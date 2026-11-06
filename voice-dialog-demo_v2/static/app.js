/* 小云语音 Demo v2 前端
 * 采集：麦克风一直上送（含播报中），echoCancellation: true
 * 播放：TTS → Web Audio → pc1→pc2 loopback → <audio>，让浏览器 AEC 把 TTS 当远端参考
 * 播完：play_end 通知服务端（合成完 ≠ 扬声器播完）
 */
(() => {
  const $ = (id) => document.getElementById(id);
  const statusEl = $("status"), transcriptEl = $("transcript"),
        metricsEl = $("metrics"), btn = $("btnStart");
  const waveEl = $("wave"), waveCtx2d = waveEl.getContext("2d");

  let ws = null, running = false;
  let capCtx = null, capStream = null, capNode = null;
  const playEl = $("playEl");
  let playCtx = null, playRate = 24000, playNode = null, playQueued = 0;
  let playPending = [];
  let expectPlayEnd = false;
  let masterGain = null, playDest = null, dropAudio = false;
  let pc1 = null, pc2 = null, loopSender = null;
  let echoFloor = 0.02, bargeSince = 0, playMuted = false;
  const BARGE_MIN_RMS = 0.05, BARGE_HOLD_MS = 40;
  let userBubble = null, botBubble = null, lastUserBubble = null;

  const WAVE_LEN = 4096;
  const waveBuf = new Float32Array(WAVE_LEN);
  let wavePos = 0, lastSpeakAt = 0, waveRaf = 0;
  const SPEAK_RMS = 0.02, SPEAK_HANG_MS = 400;

  function feedWave(int16) {
    let sum = 0;
    for (let i = 0; i < int16.length; i++) {
      const s = int16[i] / 32768;
      waveBuf[wavePos] = s;
      wavePos = (wavePos + 1) % WAVE_LEN;
      sum += s * s;
    }
    const rms = Math.sqrt(sum / int16.length);
    if (rms > SPEAK_RMS) lastSpeakAt = performance.now();
    return rms;
  }

  function drawWave() {
    if (!waveRaf) return;
    const speaking = performance.now() - lastSpeakAt < SPEAK_HANG_MS;
    waveEl.classList.toggle("show", speaking);
    const w = waveEl.width, h = waveEl.height, mid = h / 2;
    waveCtx2d.clearRect(0, 0, w, h);
    if (speaking) {
      waveCtx2d.fillStyle = "#4ade80";
      const bars = 96, step = WAVE_LEN / bars, bw = w / bars;
      for (let i = 0; i < bars; i++) {
        const idx = (wavePos - Math.floor((bars - i) * step) + WAVE_LEN * 2) % WAVE_LEN;
        const amp = Math.min(1, Math.abs(waveBuf[idx]) * 3);
        const bh = Math.max(2, amp * (h - 8));
        waveCtx2d.fillRect(i * bw + bw * 0.2, mid - bh / 2, bw * 0.6, bh);
      }
    }
    waveRaf = requestAnimationFrame(drawWave);
  }

  const WORKLET_SRC = [
    "class PcmCapture extends AudioWorkletProcessor {",
    "  constructor(){ super(); this.buf = []; this.len = 0; }",
    "  process(inputs){",
    "    const ch = inputs[0] && inputs[0][0];",
    "    if (ch) {",
    "      this.buf.push(ch.slice(0)); this.len += ch.length;",
    "      if (this.len >= 1024) {",
    "        const pcm = new Int16Array(this.len);",
    "        let o = 0;",
    "        for (const b of this.buf) for (let i = 0; i < b.length; i++) {",
    "          const s = Math.max(-1, Math.min(1, b[i]));",
    "          pcm[o++] = s < 0 ? s * 32768 : s * 32767;",
    "        }",
    "        this.port.postMessage(pcm.buffer, [pcm.buffer]);",
    "        this.buf = []; this.len = 0;",
    "      }",
    "    }",
    "    return true;",
    "  }",
    "}",
    "registerProcessor('pcm-capture', PcmCapture);"
  ].join("\n");

  const PLAY_WORKLET = [
    "class PcmPlay extends AudioWorkletProcessor {",
    "  constructor(){ super(); this.q=[]; this.off=0; this.queued=0;",
    "    this.starved=true; this.fade=0; this.fadeN=Math.round(sampleRate*0.02);",
    "    this.port.onmessage=(e)=>{",
    "      if(e.data==='flush'){ this.q=[]; this.off=0; this.queued=0; this.starved=true; this.fade=0; this.port.postMessage(0); return; }",
    "      const a=new Float32Array(e.data); this.q.push(a); this.queued+=a.length;",
    "    };",
    "  }",
    "  process(_i,outputs){",
    "    const out=outputs[0]&&outputs[0][0]; if(!out) return true;",
    "    let i=0;",
    "    while(i<out.length){",
    "      if(!this.q.length){ out.fill(0,i); this.starved=true; break; }",
    "      if(this.starved){ this.fade=0; this.starved=false; }",
    "      const c=this.q[0];",
    "      const n=Math.min(out.length-i, c.length-this.off);",
    "      if(this.fade<this.fadeN){",
    "        for(let k=0;k<n;k++){",
    "          const w=this.fade<this.fadeN?(this.fade/this.fadeN):1;",
    "          out[i+k]=c[this.off+k]*w; this.fade++;",
    "        }",
    "      } else {",
    "        out.set(c.subarray(this.off, this.off+n), i);",
    "      }",
    "      this.off+=n; i+=n; this.queued-=n;",
    "      if(this.off>=c.length){ this.q.shift(); this.off=0; }",
    "    }",
    "    this.port.postMessage(this.queued<0?0:this.queued);",
    "    return true;",
    "  }",
    "}",
    "registerProcessor('pcm-play', PcmPlay);"
  ].join("\n");

  function setStatus(text, on) {
    statusEl.textContent = text;
    statusEl.className = on ? "on" : "";
  }

  function addBubble(cls, text) {
    const d = document.createElement("div");
    d.className = "msg " + cls;
    d.textContent = text;
    transcriptEl.appendChild(d);
    transcriptEl.scrollTop = transcriptEl.scrollHeight;
    return d;
  }

  function foldPreview(full) {
    const one = (full || "").replace(/\s+/g, " ").trim();
    const preview = one.length > 16 ? one.slice(0, 16) + "…" : one;
    return "▸ 未回复  " + preview;
  }

  function foldUserBubble(el) {
    if (!el || el.dataset.full != null) return;
    const full = el.textContent;
    el.dataset.full = full;
    el.classList.add("folded", "expandable");
    el.textContent = foldPreview(full);
    el.title = "点击展开";
    el.addEventListener("click", () => {
      if (el.classList.contains("folded")) {
        el.classList.remove("folded");
        el.textContent = el.dataset.full;
        el.title = "点击收起";
      } else {
        el.classList.add("folded");
        el.textContent = foldPreview(el.dataset.full);
        el.title = "点击展开";
      }
    });
  }

  function stopPlayback() {
    if (playEl) {
      playEl.volume = 0;
      try { playEl.pause(); } catch (e) {}
    }
    if (masterGain && playCtx) {
      try {
        const t = playCtx.currentTime;
        masterGain.gain.cancelScheduledValues(t);
        masterGain.gain.setValueAtTime(0, t);
      } catch (e) {}
    }
    if (playNode) {
      try { playNode.port.postMessage("flush"); } catch (e) {}
    }
    playQueued = 0;
    playPending = [];
  }

  function mutePlayback() {
    playMuted = true;
    if (playEl) playEl.volume = 0;
    if (masterGain && playCtx) {
      try {
        const t = playCtx.currentTime;
        masterGain.gain.cancelScheduledValues(t);
        masterGain.gain.setValueAtTime(0, t);
      } catch (e) {}
    }
  }

  function unmutePlayback() {
    playMuted = false;
    dropAudio = false;
    if (masterGain && playCtx) {
      try { masterGain.gain.setValueAtTime(1, playCtx.currentTime); } catch (e) {}
    }
    if (playEl) {
      playEl.volume = 1;
      const p = playEl.play();
      if (p && p.catch) p.catch(() => {});
    }
  }

  function closeLoopback() {
    loopSender = null;
    if (pc1) { try { pc1.close(); } catch (e) {} pc1 = null; }
    if (pc2) { try { pc2.close(); } catch (e) {} pc2 = null; }
  }

  async function negotiate() {
    const offer = await pc1.createOffer();
    await pc1.setLocalDescription(offer);
    await pc2.setRemoteDescription(pc1.localDescription);
    const answer = await pc2.createAnswer();
    await pc2.setLocalDescription(answer);
    await pc1.setRemoteDescription(pc2.localDescription);
  }

  async function setupLoopback() {
    closeLoopback();
    pc1 = new RTCPeerConnection();
    pc2 = new RTCPeerConnection();
    pc1.onicecandidate = (e) => { if (e.candidate) pc2.addIceCandidate(e.candidate); };
    pc2.onicecandidate = (e) => { if (e.candidate) pc1.addIceCandidate(e.candidate); };
    pc2.ontrack = (e) => {
      const stream = e.streams[0] || new MediaStream([e.track]);
      if (playEl.srcObject !== stream) playEl.srcObject = stream;
      playEl.volume = playMuted ? 0 : 1;
      const p = playEl.play();
      if (p && p.catch) p.catch(() => {});
    };
  }

  async function ensurePlayGraph(rate) {
    playRate = rate;
    const reuse = playCtx && playCtx.sampleRate === rate && playDest && loopSender;
    if (!reuse) {
      const oldCtx = playCtx;
      playCtx = new AudioContext({ sampleRate: rate });
      masterGain = playCtx.createGain();
      playDest = playCtx.createMediaStreamDestination();
      masterGain.connect(playDest);
      playNode = null;
      const track = playDest.stream.getAudioTracks()[0];
      if (!pc1) await setupLoopback();
      if (loopSender) {
        await loopSender.replaceTrack(track);
      } else {
        loopSender = pc1.addTrack(track, playDest.stream);
        await negotiate();
      }
      if (oldCtx && oldCtx !== playCtx) {
        try { await oldCtx.close(); } catch (e) {}
      }
    }
    if (!playNode && playCtx) {
      const url = URL.createObjectURL(new Blob([PLAY_WORKLET], { type: "application/javascript" }));
      await playCtx.audioWorklet.addModule(url);
      playNode = new AudioWorkletNode(playCtx, "pcm-play");
      playNode.connect(masterGain);
      playNode.port.onmessage = (e) => {
        playQueued = e.data | 0;
        maybeSendPlayEnd();
      };
      if (playPending.length) {
        for (const a of playPending) playNode.port.postMessage(a.buffer, [a.buffer]);
        playPending = [];
      }
    }
  }

  function attachPlayEl() {
    if (!playEl) return;
    playEl.volume = playMuted ? 0 : 1;
    const p = playEl.play();
    if (p && p.catch) p.catch(() => {});
  }

  function isPlaying() {
    if (dropAudio || playMuted) return false;
    return playQueued > playRate * 0.02;
  }

  function maybeSendPlayEnd() {
    if (expectPlayEnd && playQueued <= 0 && ws && ws.readyState === 1) {
      expectPlayEnd = false;
      ws.send(JSON.stringify({ type: "play_end" }));
    }
  }

  function playChunk(bytes) {
    if (dropAudio || !playCtx || !masterGain) return;
    if (!playMuted && masterGain.gain.value < 0.5) {
      masterGain.gain.setValueAtTime(1, playCtx.currentTime);
    }
    attachPlayEl();
    const pcm = new Int16Array(bytes);
    const f32 = new Float32Array(pcm.length);
    for (let i = 0; i < pcm.length; i++) f32[i] = pcm[i] / 32768;
    playQueued += f32.length;
    if (!playNode) {
      playPending.push(f32);
      ensurePlayGraph(playRate);
      return;
    }
    playNode.port.postMessage(f32.buffer, [f32.buffer]);
  }

  const HANDLERS = {
    state(m) {
      const map = { IDLE: "待机中", LISTENING: "正在聆听…", THINKING: "思考中…", SPEAKING: "播报中（可打断）" };
      setStatus(map[m.value] || m.value, true);
    },
    asr_partial(m) {
      if (!userBubble) userBubble = addBubble("user partial", "");
      userBubble.textContent = m.text;
      transcriptEl.scrollTop = transcriptEl.scrollHeight;
    },
    asr_final(m) {
      let el = userBubble;
      if (el) {
        el.classList.remove("partial");
        el.textContent = m.text || "（未识别）";
      } else if (m.text) {
        el = addBubble("user", m.text);
      }
      lastUserBubble = el;
      userBubble = null;
    },
    ignored() {
      if (lastUserBubble) foldUserBubble(lastUserBubble);
    },
    bot_start(m) {
      dropAudio = false;
      playMuted = false;
      echoFloor = 0.02;
      bargeSince = 0;
      stopPlayback();
      expectPlayEnd = false;
      playRate = m.rate || 24000;
      if (!playCtx || playCtx.sampleRate !== playRate) {
        ensurePlayGraph(playRate).then(() => {
          if (masterGain && playCtx) masterGain.gain.setValueAtTime(1, playCtx.currentTime);
          if (playCtx && playCtx.state === "suspended") playCtx.resume();
          attachPlayEl();
        });
      } else {
        if (masterGain) masterGain.gain.setValueAtTime(1, playCtx.currentTime);
        if (playCtx.state === "suspended") playCtx.resume();
        attachPlayEl();
      }
      botBubble = addBubble("bot", "");
    },
    llm_delta(m) {
      if (!botBubble) botBubble = addBubble("bot", "");
      botBubble.textContent += m.text;
      transcriptEl.scrollTop = transcriptEl.scrollHeight;
    },
    llm_done() { botBubble = null; },
    interrupted() {
      dropAudio = true;
      playMuted = false;
      bargeSince = 0;
      stopPlayback();
      expectPlayEnd = false;
      botBubble = null;
      addBubble("sys", "—— 已打断 ——");
    },
    barge_cancel() {
      unmutePlayback();
      bargeSince = 0;
    },
    metrics(m) {
      metricsEl.textContent =
        `VAD→ASR ${m.vad_end_to_asr_ms ?? "-"}ms | ASR→LLM首token ${m.asr_to_llm_first_ms ?? "-"}ms | ` +
        `LLM→首音频 ${m.llm_first_to_audio_ms ?? "-"}ms | TTS首块 ${m.tts_first_chunk_ms ?? "-"}ms | ` +
        `端到端(说完→出声) ${m.total_e2e_ms ?? "-"}ms`;
    },
    error(m) { addBubble("sys", `[错误@${m.where}] ${m.msg}`); },
    bot_end() { expectPlayEnd = true; maybeSendPlayEnd(); },
  };

  function connect() {
    return new Promise((resolve, reject) => {
      const proto = location.protocol === "https:" ? "wss" : "ws";
      ws = new WebSocket(`${proto}://${location.host}/ws`);
      ws.binaryType = "arraybuffer";
      ws.onopen = resolve;
      ws.onerror = reject;
      ws.onclose = () => { setStatus("连接已断开", false); if (running) stop(); };
      ws.onmessage = (ev) => {
        if (typeof ev.data === "string") {
          const m = JSON.parse(ev.data);
          (HANDLERS[m.type] || (() => {}))(m);
        } else {
          playChunk(ev.data);
        }
      };
    });
  }

  async function start() {
    btn.disabled = true;
    try {
      await connect();
      capStream = await navigator.mediaDevices.getUserMedia({
        audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true, channelCount: 1 },
      });
      capCtx = new AudioContext({ sampleRate: 16000 });
      const url = URL.createObjectURL(new Blob([WORKLET_SRC], { type: "application/javascript" }));
      await capCtx.audioWorklet.addModule(url);
      const src = capCtx.createMediaStreamSource(capStream);
      capNode = new AudioWorkletNode(capCtx, "pcm-capture");
      capNode.port.onmessage = (ev) => {
        const pcm = new Int16Array(ev.data);
        const rms = feedWave(pcm);
        if (isPlaying()) {
          echoFloor = echoFloor * 0.92 + rms * 0.08;
          const th = Math.max(BARGE_MIN_RMS, echoFloor * 3);
          if (rms > th) {
            if (!bargeSince) bargeSince = performance.now();
            if (performance.now() - bargeSince >= BARGE_HOLD_MS) {
              bargeSince = 0;
              mutePlayback();
              if (ws && ws.readyState === 1) ws.send(JSON.stringify({ type: "barge" }));
            }
          } else {
            bargeSince = 0;
          }
        } else {
          bargeSince = 0;
        }
        if (ws && ws.readyState === 1) ws.send(ev.data);
      };
      src.connect(capNode);
      await setupLoopback();
      await ensurePlayGraph(24000);
      running = true;
      waveRaf = requestAnimationFrame(drawWave);
      btn.textContent = "结束对话";
      btn.classList.add("stop");
      setStatus("已连接", true);
    } catch (e) {
      addBubble("sys", "启动失败: " + e.message);
      await stop();
    } finally {
      btn.disabled = false;
    }
  }

  async function stop() {
    running = false;
    if (waveRaf) { cancelAnimationFrame(waveRaf); waveRaf = 0; }
    waveEl.classList.remove("show");
    lastSpeakAt = 0;
    if (capNode) { capNode.disconnect(); capNode = null; }
    if (capCtx) { await capCtx.close(); capCtx = null; }
    if (capStream) { capStream.getTracks().forEach((t) => t.stop()); capStream = null; }
    stopPlayback();
    closeLoopback();
    if (playEl) { playEl.srcObject = null; }
    if (playCtx) { await playCtx.close(); playCtx = null; }
    playNode = null;
    playQueued = 0;
    playPending = [];
    masterGain = null;
    playDest = null;
    dropAudio = false;
    if (ws) { ws.close(); ws = null; }
    btn.textContent = "开始对话";
    btn.classList.remove("stop");
    setStatus("未连接", false);
  }

  btn.onclick = () => (running ? stop() : start());
})();
