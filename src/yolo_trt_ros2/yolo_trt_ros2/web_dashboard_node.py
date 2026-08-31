#!/usr/bin/env python3
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import PointStamped, PoseStamped
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import String

from detector_msgs.msg import Object2DArray, Object3DArray, RobotInspectionStatus


INDEX_HTML = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Inspection Perception Dashboard</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #101418;
      --panel: #171d22;
      --panel-2: #1f272e;
      --line: #33404a;
      --text: #e6edf3;
      --muted: #9ba8b3;
      --ok: #33d17a;
      --warn: #f6d365;
      --bad: #ff6b6b;
      --accent: #6cb6ff;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font: 14px/1.45 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    header {
      height: 52px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 0 18px;
      border-bottom: 1px solid var(--line);
      background: #0d1115;
    }
    h1 {
      margin: 0;
      font-size: 16px;
      font-weight: 650;
      letter-spacing: 0;
    }
    main {
      display: grid;
      grid-template-columns: minmax(420px, 1.45fr) minmax(360px, 0.9fr);
      gap: 14px;
      padding: 14px;
      min-height: calc(100vh - 52px);
    }
    section {
      min-width: 0;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
    }
    .video-wrap {
      display: grid;
      grid-template-rows: auto 1fr;
    }
    .section-head {
      height: 42px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      padding: 0 12px;
      border-bottom: 1px solid var(--line);
      background: var(--panel-2);
      color: var(--muted);
      font-size: 12px;
    }
    .status {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      white-space: nowrap;
    }
    .pick-controls {
      display: inline-flex;
      align-items: center;
      gap: 8px;
    }
    .pick-button {
      padding: 5px 10px;
      border: 1px solid var(--accent);
      border-radius: 5px;
      background: transparent;
      color: var(--accent);
      cursor: pointer;
      font: inherit;
    }
    .pick-button.active {
      background: var(--accent);
      color: #071019;
      font-weight: 700;
    }
    .dot {
      width: 9px;
      height: 9px;
      border-radius: 50%;
      background: var(--bad);
    }
    .dot.ok { background: var(--ok); }
    .dot.warn { background: var(--warn); }
    .preview {
      position: relative;
      width: 100%;
      min-height: 360px;
      background: #050708;
      overflow: hidden;
    }
    .depth-wrap {
      border-top: 1px solid var(--line);
      background: #050708;
    }
    .depth-preview {
      position: relative;
      width: min(480px, 42%);
      max-width: 100%;
      margin: 0;
      background: #050708;
      overflow: hidden;
      cursor: crosshair;
    }
    .depth-preview .video {
      width: 100%;
      height: auto;
      display: block;
      image-rendering: auto;
    }
    #depthTooltip {
      position: absolute;
      z-index: 16;
      display: none;
      min-width: 140px;
      padding: 6px 8px;
      border: 1px solid var(--accent);
      border-radius: 6px;
      background: rgba(13,17,21,.96);
      color: var(--text);
      white-space: pre-wrap;
      pointer-events: none;
      font-variant-numeric: tabular-nums;
      font-size: 12px;
      line-height: 1.35;
    }
    .preview.pick-mode,
    .preview.pick-mode .bbox,
    .preview.pick-mode .grasp-point,
    .preview.pick-mode .mid-air-point {
      cursor: crosshair;
    }
    .pick-marker {
      position: absolute;
      z-index: 18;
      width: 22px;
      height: 22px;
      margin-left: -11px;
      margin-top: -11px;
      border: 2px solid #ffffff;
      border-radius: 50%;
      box-shadow: 0 0 0 2px #ff365f;
      pointer-events: none;
    }
    .video {
      width: 100%;
      height: auto;
      display: block;
    }
    #overlay {
      position: absolute;
      inset: 0;
      pointer-events: none;
    }
    #warning {
      position: absolute;
      left: 14px;
      top: 14px;
      right: 14px;
      z-index: 12;
      display: none;
      padding: 10px 12px;
      border: 1px solid var(--warn);
      border-radius: 6px;
      background: rgba(13,17,21,.9);
      color: var(--warn);
      white-space: pre-wrap;
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      pointer-events: none;
    }
    .bbox {
      position: absolute;
      border: 2px solid var(--ok);
      box-shadow: 0 0 0 1px rgba(0,0,0,.55);
      pointer-events: auto;
      cursor: copy;
    }
    .bbox.unstable {
      border-color: var(--warn);
      cursor: default;
    }
    .bbox-label {
      position: absolute;
      left: 0;
      top: -22px;
      max-width: 280px;
      padding: 2px 6px;
      background: rgba(0,0,0,.76);
      color: var(--text);
      font-size: 12px;
      line-height: 18px;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    .grasp-point {
      position: absolute;
      width: 12px;
      height: 12px;
      margin-left: -6px;
      margin-top: -6px;
      border: 2px solid #ffdf4d;
      background: #ff365f;
      box-sizing: border-box;
      pointer-events: auto;
      cursor: copy;
    }
    .grasp-center {
      position: absolute;
      width: 22px;
      height: 22px;
      margin-left: -11px;
      margin-top: -11px;
      color: #ff365f;
      font-size: 22px;
      line-height: 22px;
      font-weight: 800;
      text-align: center;
      pointer-events: none;
    }
    .mid-air-point {
      position: absolute;
      width: 14px;
      height: 14px;
      margin-left: -7px;
      margin-top: -7px;
      border: 2px solid #ffffff;
      background: #00d4ff;
      box-sizing: border-box;
      pointer-events: auto;
      cursor: copy;
    }
    #tooltip {
      position: fixed;
      z-index: 20;
      display: none;
      max-width: 440px;
      padding: 8px 10px;
      border: 1px solid var(--accent);
      border-radius: 6px;
      background: rgba(13,17,21,.96);
      color: var(--text);
      white-space: pre-wrap;
      pointer-events: none;
      font-variant-numeric: tabular-nums;
    }
    .side {
      display: grid;
      gap: 14px;
      align-content: start;
    }
    .content {
      padding: 12px;
    }
    table {
      width: 100%;
      border-collapse: collapse;
    }
    th, td {
      text-align: left;
      padding: 7px 6px;
      border-bottom: 1px solid rgba(255,255,255,0.07);
      vertical-align: top;
      font-variant-numeric: tabular-nums;
    }
    th {
      color: var(--muted);
      font-size: 12px;
      font-weight: 600;
    }
    .kv {
      display: grid;
      grid-template-columns: 128px 1fr;
      gap: 8px 12px;
      font-variant-numeric: tabular-nums;
    }
    .key { color: var(--muted); }
    .mono {
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      word-break: break-word;
    }
    .copy { color: var(--accent); }
    .empty {
      color: var(--muted);
      padding: 12px 0;
    }
    @media (max-width: 920px) {
      main { grid-template-columns: 1fr; }
      .video { min-height: 260px; }
    }
  </style>
</head>
<body>
  <header>
    <h1>Inspection Perception Dashboard</h1>
    <div class="status"><span id="copy" class="copy"></span><span id="dot" class="dot"></span><span id="status">connecting</span></div>
  </header>
  <main>
    <section class="video-wrap">
      <div class="section-head">
        <span>debug image stream</span>
        <span class="pick-controls">
          <button id="pick3dButton" class="pick-button" type="button">选取3D点 (X)</button>
          <span id="imageMeta" class="mono">--</span>
        </span>
      </div>
      <div id="preview" class="preview">
        <img id="stream" class="video" src="/stream.mjpg" alt="debug image stream">
        <div id="overlay"></div>
        <div id="warning"></div>
      </div>
      <div class="depth-wrap">
        <div class="section-head">
          <span>depth preview</span>
          <span id="depthMeta" class="mono">hover for depth (m / mm)</span>
        </div>
        <div id="depthPreview" class="depth-preview">
          <img id="depthStream" class="video" src="/depth.mjpg" alt="depth preview stream">
          <div id="depthTooltip"></div>
        </div>
      </div>
    </section>
    <div class="side">
      <section>
        <div class="section-head"><span>2D detections</span><span id="objectMeta" class="mono">--</span></div>
        <div class="content">
          <table>
            <thead><tr><th>class</th><th>conf</th><th>bbox</th><th>center</th></tr></thead>
            <tbody id="objects2d"></tbody>
          </table>
        </div>
      </section>
      <section>
        <div class="section-head"><span>3D target</span><span id="targetMeta" class="mono">--</span></div>
        <div class="content">
          <div id="target3d" class="kv"></div>
        </div>
      </section>
      <section>
        <div class="section-head"><span>current XYZ</span><span id="currentXyzMeta" class="mono">--</span></div>
        <div class="content">
          <div id="currentXyz" class="kv" title="click to copy current XYZ"></div>
        </div>
      </section>
      <section>
        <div class="section-head"><span>IK joint state</span><span id="jointMeta" class="mono">--</span></div>
        <div class="content">
          <table>
            <thead><tr><th>joint</th><th>position rad</th></tr></thead>
            <tbody id="joints"></tbody>
          </table>
        </div>
      </section>
    </div>
  </main>
  <div id="tooltip"></div>
  <script>
    const fmt = (v, digits = 3) => Number.isFinite(v) ? Number(v).toFixed(digits) : "--";
    const stamp = h => h && h.stamp ? `${h.stamp.sec}.${String(h.stamp.nanosec).padStart(9, "0")}` : "--";
    const tracks = new Map();
    let latestData = null;
    let pick3dMode = false;
    let pendingPickRequestId = null;
    let handledPickRequestId = null;
    let lastPickedPixel = null;
    let depthHoverTimer = null;
    let lastDepthSampleKey = "";

    function iou(a, b) {
      const ix1 = Math.max(a[0], b[0]);
      const iy1 = Math.max(a[1], b[1]);
      const ix2 = Math.min(a[2], b[2]);
      const iy2 = Math.min(a[3], b[3]);
      const inter = Math.max(0, ix2 - ix1) * Math.max(0, iy2 - iy1);
      const areaA = Math.max(1, (a[2] - a[0]) * (a[3] - a[1]));
      const areaB = Math.max(1, (b[2] - b[0]) * (b[3] - b[1]));
      return inter / Math.max(1, areaA + areaB - inter);
    }

    function updateTracks(objects) {
      const now = Date.now();
      const used = new Set();
      objects.forEach((obj, index) => {
        const box = obj.bbox_xyxy || [0, 0, 0, 0];
        let bestKey = null;
        let bestIou = 0;
        for (const [key, tr] of tracks.entries()) {
          if (used.has(key) || tr.className !== obj.class_name) continue;
          const score = iou(box, tr.box);
          if (score > bestIou) {
            bestIou = score;
            bestKey = key;
          }
        }
        if (bestKey && bestIou >= 0.45) {
          const tr = tracks.get(bestKey);
          tr.box = box;
          tr.lastSeen = now;
          obj.stable_ms = now - tr.firstSeen;
          obj.track_key = bestKey;
          used.add(bestKey);
        } else {
          const key = `${obj.class_name}:${index}:${now}`;
          tracks.set(key, { className: obj.class_name, box, firstSeen: now, lastSeen: now });
          obj.stable_ms = 0;
          obj.track_key = key;
          used.add(key);
        }
      });
      for (const [key, tr] of tracks.entries()) {
        if (now - tr.lastSeen > 1500) tracks.delete(key);
      }
    }

    function coordText(obj) {
      const p = obj.point_torso_m;
      if (!p) return "world unavailable";
      return p.map(v => fmt(v, 4)).join(" ");
    }

    function xyzText(values) {
      if (!values || values.length !== 3) return "";
      return values.map(v => fmt(v, 4)).join(" ");
    }

    function isBluePress(obj) {
      const name = String(
        (obj && (obj.class_name || (obj.detection && obj.detection.class_name))) || ""
      ).toLowerCase();
      return name.includes("blue") ||
        name.includes("circle push point") ||
        name.includes("white square push point") ||
        name.includes("red sticker push point") ||
        name.includes("lock point");
    }

    function isWorkTagGrasp(obj) {
      const name = String(
        (obj && (obj.class_name || (obj.detection && obj.detection.class_name))) || ""
      ).toLowerCase();
      return name.includes("work tag grasp");
    }

    function isHangHook(obj) {
      const name = String(
        (obj && (obj.class_name || (obj.detection && obj.detection.class_name))) || ""
      ).toLowerCase();
      return name.includes("cabinet hang hook");
    }

    function ikText(obj) {
      const ik = obj.ik;
      if (!ik || !ik.joint_values_rad) return "ik unavailable";
      return Object.entries(ik.joint_values_rad).map(([k, v]) => `${k}: ${fmt(v, 5)}`).join("\\n");
    }

    async function copyText(text) {
      text = String(text)
        .replace(/[\\[\\],]/g, " ")
        .trim()
        .replace(/\\s+/g, " ");
      if (navigator.clipboard && window.isSecureContext) {
        await navigator.clipboard.writeText(text);
        return;
      }
      const area = document.createElement("textarea");
      area.value = text;
      area.style.position = "fixed";
      area.style.left = "-9999px";
      document.body.appendChild(area);
      area.focus();
      area.select();
      document.execCommand("copy");
      document.body.removeChild(area);
    }

    function setPick3dMode(enabled) {
      pick3dMode = Boolean(enabled);
      document.getElementById("preview").classList.toggle("pick-mode", pick3dMode);
      const button = document.getElementById("pick3dButton");
      button.classList.toggle("active", pick3dMode);
      button.textContent = pick3dMode ? "点击画面取点 (Esc取消)" : "选取3D点 (X)";
    }

    async function handlePick3dResult(result) {
      if (!result || !Number.isFinite(Number(result.request_id))) return;
      const requestId = Number(result.request_id);
      if (handledPickRequestId === requestId) return;
      handledPickRequestId = requestId;
      pendingPickRequestId = null;
      const copy = document.getElementById("copy");
      if (!result.valid || !result.point_world) {
        copy.textContent = `3D取点失败: ${result.message || "unknown error"}`;
        setTimeout(() => { copy.textContent = ""; }, 3000);
        return;
      }
      lastPickedPixel = result.pixel;
      const text = xyzText(result.point_world);
      await copyText(text);
      copy.textContent =
        `copied foreground ${text} | R_ee ${xyzText(result.point_target)} | depth ${fmt(result.depth_m, 4)}`;
      setTimeout(() => { copy.textContent = ""; }, 4000);
      if (latestData) renderOverlay(latestData);
    }

    async function queryPick3d(u, v) {
      const copy = document.getElementById("copy");
      copy.textContent = `querying pixel ${fmt(u, 1)} ${fmt(v, 1)} ...`;
      try {
        const response = await fetch("/api/pick3d", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ u, v })
        });
        const result = await response.json();
        if (!result.accepted) {
          copy.textContent = `3D取点失败: ${result.message || "request rejected"}`;
          return;
        }
        pendingPickRequestId = Number(result.request_id);
        if (!result.pending) await handlePick3dResult(result);
      } catch (err) {
        copy.textContent = `3D取点请求失败: ${err}`;
      }
    }

    function renderOverlay(data) {
      const overlay = document.getElementById("overlay");
      const stream = document.getElementById("stream");
      const warning = document.getElementById("warning");
      overlay.innerHTML = "";
      const diagnostics = data.diagnostics || [];
      if (diagnostics.length) {
        warning.style.display = "block";
        warning.textContent = diagnostics.join("\\n");
      } else {
        warning.style.display = "none";
        warning.textContent = "";
      }
      const objects = (data.objects || []).filter(o => o && o.bbox_xyxy);
      updateTracks(objects);
      const info = data.image || {};
      const srcW = info.width || stream.naturalWidth || 1280;
      const srcH = info.height || stream.naturalHeight || 720;
      const rect = stream.getBoundingClientRect();
      const sx = rect.width / srcW;
      const sy = rect.height / srcH;

      objects.forEach(obj => {
        const box = obj.bbox_xyxy;
        const stable = (obj.stable_ms || 0) >= 3000;
        const div = document.createElement("div");
        div.className = "bbox" + (stable ? "" : " unstable");
        div.style.left = `${box[0] * sx}px`;
        div.style.top = `${box[1] * sy}px`;
        div.style.width = `${Math.max(4, (box[2] - box[0]) * sx)}px`;
        div.style.height = `${Math.max(4, (box[3] - box[1]) * sy)}px`;

        const label = document.createElement("div");
        label.className = "bbox-label";
        label.textContent = `${obj.object_id || ""} ${obj.semantic_name || obj.class_name} ${fmt(obj.confidence, 2)}`;
        div.appendChild(label);

        div.addEventListener("mousemove", ev => {
          const tooltip = document.getElementById("tooltip");
          if (stable) {
            const copyHint = obj.seg_geometric_center_target_m
              ? "click to copy SEG geometric center"
              : (isBluePress(obj)
                ? "click to copy blue press target"
                : ((obj.handle_mid_right_air_target_m || obj.handle_grasp_ree_target_m)
                    ? "click to copy explicit handle grasp target"
                    : (obj.point_torso_m
                        ? "double click to copy object center"
                        : "3D point unavailable")));
            tooltip.textContent =
              `${label.textContent}\\nworld ${coordText(obj)}\\n${obj.message || ""}\\n${ikText(obj)}\\n${copyHint}`;
          } else {
            tooltip.textContent =
              `${label.textContent}\\nstabilizing ${fmt((obj.stable_ms || 0) / 1000, 1)}/3.0s`;
          }
          tooltip.style.display = "block";
          tooltip.style.left = `${ev.clientX + 14}px`;
          tooltip.style.top = `${ev.clientY + 14}px`;
        });
        div.addEventListener("mouseleave", () => {
          document.getElementById("tooltip").style.display = "none";
        });
        div.addEventListener("click", async () => {
          if (!stable) return;
          let text = "";
          let kind = "";
          if (obj.seg_geometric_center_target_m) {
            text = xyzText(obj.seg_geometric_center_target_m);
            kind = "SEG geometric center";
          } else if (isBluePress(obj) && obj.point_torso_m) {
            text = coordText(obj);
            kind = "press target";
          } else {
            const handleTarget = obj.handle_mid_right_air_target_m || obj.handle_grasp_ree_target_m;
            if (handleTarget) {
              text = xyzText(handleTarget);
              kind = "handle target";
            } else if (obj.point_torso_m) {
              text = coordText(obj);
              kind = "object center";
            } else {
              return;
            }
          }
          await copyText(text);
          const copy = document.getElementById("copy");
          copy.textContent = `copied ${kind} ${text}`;
          setTimeout(() => { copy.textContent = ""; }, 1800);
        });
        div.addEventListener("dblclick", async ev => {
          ev.stopPropagation();
          let text = "";
          let targetKind = "";
          if (obj.seg_geometric_center_target_m) {
            text = xyzText(obj.seg_geometric_center_target_m);
            targetKind = "SEG geometric center";
          } else if (isBluePress(obj) && obj.point_torso_m) {
            text = coordText(obj);
            targetKind = "press target";
          } else {
            const target = obj.handle_mid_right_air_target_m || obj.handle_grasp_ree_target_m;
            if (target) {
              text = xyzText(target);
              targetKind = "handle target";
            } else if (obj.point_torso_m) {
              text = coordText(obj);
              targetKind = "object center";
            } else {
              return;
            }
          }
          await copyText(text);
          const copy = document.getElementById("copy");
          copy.textContent = `copied ${targetKind} ${text}`;
          setTimeout(() => { copy.textContent = ""; }, 1800);
        });
        overlay.appendChild(div);

        const edge = obj.handle_grasp_edge_px || [];
        const targets = obj.handle_grasp_endpoint_targets_m || [];
        edge.forEach((p, endpointIndex) => {
          if (!p || p.length !== 2) return;
          const marker = document.createElement("div");
          marker.className = "grasp-point";
          marker.style.left = `${p[0] * sx}px`;
          marker.style.top = `${p[1] * sy}px`;
          marker.addEventListener("mousemove", ev => {
            const tooltip = document.getElementById("tooltip");
            const target = targets[endpointIndex];
            tooltip.textContent =
              `${obj.object_id || ""} handle endpoint ${endpointIndex}\\npx [${fmt(p[0], 1)}, ${fmt(p[1], 1)}]\\ntarget ${target ? xyzText(target) : "unavailable"}\\ndouble click to copy`;
            tooltip.style.display = "block";
            tooltip.style.left = `${ev.clientX + 14}px`;
            tooltip.style.top = `${ev.clientY + 14}px`;
          });
          marker.addEventListener("mouseleave", () => {
            document.getElementById("tooltip").style.display = "none";
          });
          marker.addEventListener("dblclick", async ev => {
            ev.stopPropagation();
            const target = targets[endpointIndex];
            if (!target) return;
            const text = xyzText(target);
            await copyText(text);
            const copy = document.getElementById("copy");
            copy.textContent = `copied endpoint ${endpointIndex} ${text}`;
            setTimeout(() => { copy.textContent = ""; }, 1800);
          });
          overlay.appendChild(marker);
        });

        const geom = obj.geometric_center_px || [];
        if (geom.length === 2) {
          const geomMarker = document.createElement("div");
          geomMarker.className = "grasp-center";
          geomMarker.style.left = `${geom[0] * sx}px`;
          geomMarker.style.top = `${geom[1] * sy}px`;
          geomMarker.style.color = "#ffe066";
          geomMarker.textContent = "G";
          geomMarker.title = "SEG geometric center";
          overlay.appendChild(geomMarker);
        }

        const center = obj.handle_grasp_center_px || [];
        if (center.length === 2) {
          const centerMarker = document.createElement("div");
          centerMarker.className = "grasp-center";
          centerMarker.style.left = `${center[0] * sx}px`;
          centerMarker.style.top = `${center[1] * sy}px`;
          centerMarker.textContent = "+";
          overlay.appendChild(centerMarker);
        }

        const mid = obj.handle_mid_px || [];
        if (mid.length === 2) {
          const midMarker = document.createElement("div");
          midMarker.className = "mid-air-point";
          midMarker.style.left = `${mid[0] * sx}px`;
          midMarker.style.top = `${mid[1] * sy}px`;
          midMarker.addEventListener("mousemove", ev => {
            const tooltip = document.getElementById("tooltip");
            const target = obj.handle_mid_right_air_target_m;
            const world = obj.handle_mid_right_air_world_m;
            tooltip.textContent =
              `${obj.object_id || ""} handle mid-right air\\npx [${fmt(mid[0], 1)}, ${fmt(mid[1], 1)}]\\nworld ${world ? xyzText(world) : "unavailable"}\\ntarget ${target ? xyzText(target) : "unavailable"}\\ndouble click to copy`;
            tooltip.style.display = "block";
            tooltip.style.left = `${ev.clientX + 14}px`;
            tooltip.style.top = `${ev.clientY + 14}px`;
          });
          midMarker.addEventListener("mouseleave", () => {
            document.getElementById("tooltip").style.display = "none";
          });
          midMarker.addEventListener("dblclick", async ev => {
            ev.stopPropagation();
            const target = obj.handle_mid_right_air_target_m;
            if (!target) return;
            const text = xyzText(target);
            await copyText(text);
            const copy = document.getElementById("copy");
            copy.textContent = `copied mid-right air ${text}`;
            setTimeout(() => { copy.textContent = ""; }, 1800);
          });
          overlay.appendChild(midMarker);
        }
      });
      if (lastPickedPixel && lastPickedPixel.length === 2) {
        const marker = document.createElement("div");
        marker.className = "pick-marker";
        marker.style.left = `${lastPickedPixel[0] * sx}px`;
        marker.style.top = `${lastPickedPixel[1] * sy}px`;
        overlay.appendChild(marker);
      }
    }

    function setRows(tbody, rows, emptyCols) {
      tbody.innerHTML = "";
      if (!rows.length) {
        const tr = document.createElement("tr");
        const td = document.createElement("td");
        td.colSpan = emptyCols;
        td.className = "empty";
        td.textContent = "no data";
        tr.appendChild(td);
        tbody.appendChild(tr);
        return;
      }
      rows.forEach(row => tbody.appendChild(row));
    }
    function kv(el, pairs) {
      el.innerHTML = "";
      if (!pairs.length) {
        const div = document.createElement("div");
        div.className = "empty";
        div.textContent = "no data";
        el.appendChild(div);
        return;
      }
      pairs.forEach(([k, v]) => {
        const key = document.createElement("div");
        key.className = "key";
        key.textContent = k;
        const val = document.createElement("div");
        val.className = "mono";
        val.textContent = v;
        el.appendChild(key);
        el.appendChild(val);
      });
    }
    async function refresh() {
      try {
        const res = await fetch("/api/state", { cache: "no-store" });
        const data = await res.json();
        latestData = data;
        const age = data.server_time - data.last_update_time;
        const dot = document.getElementById("dot");
        dot.className = "dot " + (age < 1.5 ? "ok" : age < 5 ? "warn" : "");
        document.getElementById("status").textContent = `age ${fmt(age, 2)}s objects=${(data.objects || []).length}`;
        document.getElementById("imageMeta").textContent = `${data.image.width || "--"}x${data.image.height || "--"} q=${data.image.jpeg_quality}`;
        const depth = data.depth || {};
        const validPct = Number.isFinite(depth.valid_ratio) ? (100.0 * depth.valid_ratio).toFixed(1) : "--";
        document.getElementById("depthMeta").textContent =
          depth.width
            ? `${depth.width}x${depth.height} valid=${validPct}% scale=${fmt(depth.depth_scale, 4)} hover for m/mm`
            : "waiting for depth";
        document.getElementById("objectMeta").textContent = `${data.objects2d.objects.length} objects`;
        renderOverlay(data);
        const pickResult = data.pixel_query_result;
        if (
          pickResult &&
          pendingPickRequestId !== null &&
          Number(pickResult.request_id) !== handledPickRequestId &&
          Number(pickResult.request_id) === pendingPickRequestId
        ) {
          await handlePick3dResult(pickResult);
        }
        const objectRows = data.objects2d.objects.map(o => {
          const tr = document.createElement("tr");
          [o.semantic_name || o.class_name, fmt(o.confidence, 2), `${o.xmin},${o.ymin},${o.xmax},${o.ymax}`, `${fmt(o.cx, 1)},${fmt(o.cy, 1)}`]
            .forEach(text => {
              const td = document.createElement("td");
              td.className = "mono";
              td.textContent = text;
              tr.appendChild(td);
            });
          return tr;
        });
        setRows(document.getElementById("objects2d"), objectRows, 4);
        const valid3d = data.objects3d.objects.filter(o => o.valid);
        const best3d = valid3d.find(o => isBluePress(o))
          || valid3d.find(o => isWorkTagGrasp(o))
          || valid3d.find(o => isHangHook(o))
          || valid3d[0]
          || data.objects3d.objects[0];
        const picked3d = data.pixel_query_result;
        document.getElementById("targetMeta").textContent =
          picked3d && picked3d.valid ? (picked3d.target_frame || "--") :
          (best3d ? (best3d.target_frame || "--") : "--");
        if (picked3d && picked3d.valid) {
          kv(document.getElementById("target3d"), [
            ["kind", "selected foreground pixel"],
            ["pixel", `${fmt(picked3d.pixel[0], 1)} ${fmt(picked3d.pixel[1], 1)}`],
            ["depth_m", fmt(picked3d.depth_m, 4)],
            ["depth_source", picked3d.depth_source || "--"],
            ["point_camera", xyzText(picked3d.point_camera)],
            ["point_world (copied)", xyzText(picked3d.point_world)],
            ["R_ee target", xyzText(picked3d.point_target)],
            ["target_frame", picked3d.target_frame || "--"],
            ["message", picked3d.message || "--"]
          ]);
        } else if (best3d) {
          kv(document.getElementById("target3d"), [
            ["class", best3d.detection.class_name],
            ["confidence", fmt(best3d.detection.confidence, 3)],
            ["depth_m", fmt(best3d.depth_m, 3)],
            ["source_frame", best3d.source_frame || "--"],
            ["target_frame", best3d.target_frame || "--"],
            ["point_camera", `${fmt(best3d.point_camera.x)}, ${fmt(best3d.point_camera.y)}, ${fmt(best3d.point_camera.z)}`],
            ["point_target", `${fmt(best3d.point_target.x)}, ${fmt(best3d.point_target.y)}, ${fmt(best3d.point_target.z)}`],
            ["message", best3d.message || "--"],
            ["stamp", stamp(best3d.header)]
          ]);
        } else if (data.target_pose.header) {
          const p = data.target_pose.pose.position;
          kv(document.getElementById("target3d"), [
            ["target_frame", data.target_pose.header.frame_id || "--"],
            ["pose_xyz", `${fmt(p.x)}, ${fmt(p.y)}, ${fmt(p.z)}`],
            ["stamp", stamp(data.target_pose.header)]
          ]);
        } else {
          kv(document.getElementById("target3d"), []);
        }
        const currentEe = data.current_ee_point || {};
        const currentEePoint = currentEe.point || {};
        document.getElementById("currentXyzMeta").textContent =
          currentEe.header ? (currentEe.header.frame_id || "--") : "--";
        if (currentEe.header) {
          kv(document.getElementById("currentXyz"), [
            ["link", "right_wrist_yaw_link"],
            ["xyz", xyzText([currentEePoint.x, currentEePoint.y, currentEePoint.z])],
            ["frame", currentEe.header.frame_id || "--"],
            ["stamp", stamp(currentEe.header)],
            ["copy", "click this panel"]
          ]);
        } else {
          kv(document.getElementById("currentXyz"), []);
        }
        const jointState = data.target_joint_state.name.length ? data.target_joint_state : data.current_joint_state;
        const jointKind = data.target_joint_state.name.length ? "target" : "current";
        document.getElementById("jointMeta").textContent = `${jointKind} ${jointState.name.length} joints`;
        const jointRows = jointState.name.map((name, i) => {
          const tr = document.createElement("tr");
          [name, fmt(jointState.position[i], 5)].forEach(text => {
            const td = document.createElement("td");
            td.className = "mono";
            td.textContent = text;
            tr.appendChild(td);
          });
          return tr;
        });
        setRows(document.getElementById("joints"), jointRows, 2);
      } catch (err) {
        document.getElementById("dot").className = "dot";
        document.getElementById("status").textContent = "disconnected";
      }
    }
    window.addEventListener("resize", () => { if (latestData) renderOverlay(latestData); });
    document.getElementById("currentXyz").addEventListener("click", async () => {
      const currentEe = latestData && latestData.current_ee_point;
      if (!currentEe || !currentEe.header || !currentEe.point) return;
      const text = xyzText([currentEe.point.x, currentEe.point.y, currentEe.point.z]);
      await copyText(text);
      const copy = document.getElementById("copy");
      copy.textContent = `copied current XYZ ${text}`;
      setTimeout(() => { copy.textContent = ""; }, 1800);
    });
    document.getElementById("target3d").addEventListener("click", async () => {
      const result = latestData && latestData.pixel_query_result;
      if (!result || !result.valid || !result.point_world) return;
      const text = xyzText(result.point_world);
      await copyText(text);
      const copy = document.getElementById("copy");
      copy.textContent = `copied selected foreground ${text}`;
      setTimeout(() => { copy.textContent = ""; }, 1800);
    });
    document.getElementById("pick3dButton").addEventListener("click", () => {
      setPick3dMode(!pick3dMode);
    });
    document.getElementById("preview").addEventListener("click", ev => {
      if (!pick3dMode || !latestData) return;
      ev.preventDefault();
      ev.stopImmediatePropagation();
      const stream = document.getElementById("stream");
      const rect = stream.getBoundingClientRect();
      if (
        ev.clientX < rect.left || ev.clientX > rect.right ||
        ev.clientY < rect.top || ev.clientY > rect.bottom
      ) return;
      const srcW = latestData.image.width || stream.naturalWidth || 1280;
      const srcH = latestData.image.height || stream.naturalHeight || 720;
      const u = (ev.clientX - rect.left) * srcW / Math.max(1, rect.width);
      const v = (ev.clientY - rect.top) * srcH / Math.max(1, rect.height);
      lastPickedPixel = [u, v];
      setPick3dMode(false);
      renderOverlay(latestData);
      queryPick3d(u, v);
    }, true);
    document.addEventListener("keydown", ev => {
      if (ev.ctrlKey || ev.metaKey || ev.altKey) return;
      if (ev.key.toLowerCase() === "x") {
        ev.preventDefault();
        setPick3dMode(!pick3dMode);
      } else if (ev.key === "Escape" && pick3dMode) {
        ev.preventDefault();
        setPick3dMode(false);
      }
    });

    async function sampleDepthAtEvent(ev) {
      const img = document.getElementById("depthStream");
      const tip = document.getElementById("depthTooltip");
      const rect = img.getBoundingClientRect();
      if (rect.width <= 1 || rect.height <= 1) return;
      const nx = Math.max(0, Math.min(1, (ev.clientX - rect.left) / rect.width));
      const ny = Math.max(0, Math.min(1, (ev.clientY - rect.top) / rect.height));
      const key = nx.toFixed(3) + "," + ny.toFixed(3);
      tip.style.left = Math.min(rect.width - 8, Math.max(8, ev.clientX - rect.left + 12)) + "px";
      tip.style.top = Math.min(rect.height - 8, Math.max(8, ev.clientY - rect.top + 12)) + "px";
      tip.style.display = "block";
      if (key === lastDepthSampleKey) return;
      lastDepthSampleKey = key;
      tip.textContent = "sampling...";
      try {
        const res = await fetch("/api/depth_sample?nx=" + nx.toFixed(4) + "&ny=" + ny.toFixed(4), { cache: "no-store" });
        const data = await res.json();
        if (!data.ok) {
          tip.textContent = data.message || "no depth";
          return;
        }
        tip.textContent =
          "u=" + data.u + " v=" + data.v + "\\n" +
          "depth=" + fmt(data.depth_m, 4) + " m\\n" +
          "raw=" + data.raw_mm + " mm\\n" +
          (data.valid ? "VALID" : "HOLE / INVALID");
      } catch (err) {
        tip.textContent = "depth sample failed";
      }
    }

    document.getElementById("depthPreview").addEventListener("mousemove", (ev) => {
      if (depthHoverTimer) clearTimeout(depthHoverTimer);
      depthHoverTimer = setTimeout(() => sampleDepthAtEvent(ev), 40);
    });
    document.getElementById("depthPreview").addEventListener("mouseleave", () => {
      if (depthHoverTimer) clearTimeout(depthHoverTimer);
      document.getElementById("depthTooltip").style.display = "none";
      lastDepthSampleKey = "";
    });

    setInterval(refresh, 300);
    refresh();
  </script>
</body>
</html>
"""


def make_status_jpeg(message, width=1280, height=720):
    image = np.zeros((int(height), int(width), 3), dtype=np.uint8)
    image[:] = (5, 7, 8)
    lines = [line for part in str(message).split('\n') for line in _wrap_text(part, 72)]
    y = 70
    cv2.putText(image, 'Inspection Perception Dashboard', (36, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (230, 237, 243), 2, cv2.LINE_AA)
    for line in lines:
        cv2.putText(image, line, (36, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (80, 210, 255), 2, cv2.LINE_AA)
        y += 34
    ok, encoded = cv2.imencode('.jpg', image, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
    return encoded.tobytes() if ok else None


def _wrap_text(text, max_chars):
    text = str(text)
    if not text:
        return ['']
    return [text[i:i + max_chars] for i in range(0, len(text), max_chars)]


class DashboardState:
    def __init__(self):
        self.lock = threading.Lock()
        self.frame_condition = threading.Condition(self.lock)
        self.depth_condition = threading.Condition(self.lock)
        self.jpeg = make_status_jpeg(
            'Waiting for ROS image topic: /detector/debug_image\n'
            'If this stays here, check camera publisher and yolo_detector_node logs.'
        )
        self.image_header = None
        self.image_width = 0
        self.image_height = 0
        self.jpeg_quality = 80
        self.depth_jpeg = make_status_jpeg(
            'Waiting for depth preview\n'
            'Enabled only when webUI=true and RealSense depth is flowing.',
            width=640,
            height=480,
        )
        self.depth_raw = None
        self.depth_width = 0
        self.depth_height = 0
        self.depth_scale = 0.001
        self.depth_valid_ratio = 0.0
        self.depth_min_m = 0.10
        self.depth_max_m = 1.0
        self.last_update_time = 0.0
        self.objects2d = None
        self.objects3d = None
        self.target_point = None
        self.target_pose = None
        self.target_joint_state = None
        self.current_joint_state = None
        self.current_ee_point = None
        self.objects_ik_json = None
        self.pixel_query_result = None
        self.pixel_query_sequence = 0
        self.pixel_query_condition = threading.Condition(self.lock)
        self.robot_status = None


def colorize_depth_bgr(depth, depth_scale=0.001, min_m=0.10, max_m=1.0):
    """Render a RealSense-like depth colormap; invalid pixels stay black."""
    depth_arr = np.asarray(depth)
    if depth_arr.ndim != 2:
        raise ValueError('depth must be HxW')
    if np.issubdtype(depth_arr.dtype, np.floating):
        depth_m = depth_arr.astype(np.float32)
    else:
        depth_m = depth_arr.astype(np.float32) * float(depth_scale)
    valid = np.isfinite(depth_m) & (depth_m > 1e-6)
    vis = np.zeros(depth_m.shape, dtype=np.uint8)
    lo = float(min_m)
    hi = max(lo + 1e-3, float(max_m))
    if np.any(valid):
        clipped = np.clip(depth_m, lo, hi)
        norm = ((clipped - lo) / (hi - lo) * 255.0).astype(np.uint8)
        vis[valid] = norm[valid]
    colored = cv2.applyColorMap(vis, cv2.COLORMAP_TURBO)
    colored[~valid] = (0, 0, 0)
    valid_ratio = float(np.count_nonzero(valid)) / float(depth_m.size) if depth_m.size else 0.0
    return colored, valid_ratio


class WebDashboardNode(Node):
    """Serve a tiny browser dashboard for detector video and perception state."""

    def __init__(self, subscribe_debug_image=True):
        super().__init__('web_dashboard')
        self._declare_parameters()

        self.host = str(self.get_parameter('web_host').value)
        self.port = int(self.get_parameter('web_port').value)
        self.public_host = str(self.get_parameter('public_host').value)
        self.jpeg_quality = int(self.get_parameter('jpeg_quality').value)
        self.frame_timeout_sec = float(self.get_parameter('frame_timeout_sec').value)
        self.debug_image_topic = self.get_parameter('debug_image_topic').value
        self.objects_topic = self.get_parameter('objects_topic').value
        self.objects_3d_topic = self.get_parameter('objects_3d_topic').value
        self.target_point_topic = self.get_parameter('target_point_topic').value
        self.target_pose_topic = self.get_parameter('target_pose_topic').value
        self.target_joint_state_topic = self.get_parameter('target_joint_state_topic').value
        self.current_joint_state_topic = self.get_parameter('current_joint_state_topic').value
        self.current_ee_point_topic = self.get_parameter('current_ee_point_topic').value
        self.objects_ik_topic = self.get_parameter('objects_ik_topic').value
        self.pixel_query_topic = self.get_parameter('pixel_query_topic').value
        self.pixel_query_result_topic = self.get_parameter('pixel_query_result_topic').value
        self.robot_status_topic = self.get_parameter('robot_status_topic').value

        self.bridge = CvBridge()
        self.state = DashboardState()
        self.state.jpeg_quality = max(1, min(100, self.jpeg_quality))
        self.state.depth_scale = float(self.get_parameter('depth_scale').value)
        self.state.depth_min_m = float(self.get_parameter('depth_preview_min_m').value)
        self.state.depth_max_m = float(self.get_parameter('depth_preview_max_m').value)
        self.pixel_query_pub = self.create_publisher(PointStamped, self.pixel_query_topic, 10)

        self.debug_image_sub = None
        if subscribe_debug_image:
            self.debug_image_sub = self.create_subscription(Image, self.debug_image_topic, self._image_callback, 10)
        self.create_subscription(Object2DArray, self.objects_topic, self._objects_callback, 10)
        self.create_subscription(Object3DArray, self.objects_3d_topic, self._objects3d_callback, 10)
        self.create_subscription(PointStamped, self.target_point_topic, self._target_point_callback, 10)
        self.create_subscription(PoseStamped, self.target_pose_topic, self._target_pose_callback, 10)
        self.create_subscription(JointState, self.target_joint_state_topic, self._joint_state_callback, 10)
        self.create_subscription(JointState, self.current_joint_state_topic, self._current_joint_state_callback, 10)
        self.create_subscription(PointStamped, self.current_ee_point_topic, self._current_ee_point_callback, 10)
        self.create_subscription(String, self.objects_ik_topic, self._objects_ik_callback, 10)
        self.create_subscription(String, self.pixel_query_result_topic, self._pixel_query_result_callback, 10)
        self.create_subscription(RobotInspectionStatus, self.robot_status_topic, self._robot_status_callback, 10)

        handler_cls = self._make_handler()
        self.server = ThreadingHTTPServer((self.host, self.port), handler_cls)
        self.server.daemon_threads = True
        self.server_thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.server_thread.start()

        self.get_logger().info(
            'Web dashboard started: http://%s:%d/ image=%s objects=%s objects_3d=%s'
            % (self.public_host or self.host, self.port, self.debug_image_topic, self.objects_topic, self.objects_3d_topic)
        )

    def _declare_parameters(self):
        self.declare_parameter('web_host', '0.0.0.0')
        self.declare_parameter('web_port', 8080)
        self.declare_parameter('public_host', '192.168.25.189')
        self.declare_parameter('jpeg_quality', 80)
        self.declare_parameter('frame_timeout_sec', 2.0)
        self.declare_parameter('debug_image_topic', '/detector/debug_image')
        self.declare_parameter('objects_topic', '/detector/objects_2d')
        self.declare_parameter('objects_3d_topic', '/detector/objects')
        self.declare_parameter('target_point_topic', '/detector/target_point')
        self.declare_parameter('target_pose_topic', '/detector/target_pose')
        self.declare_parameter('target_joint_state_topic', '/detector/target_joint_state')
        self.declare_parameter('current_joint_state_topic', '/detector/current_joint_state')
        self.declare_parameter('current_ee_point_topic', '/detector/current_ee_point')
        self.declare_parameter('objects_ik_topic', '/detector/objects_ik_json')
        self.declare_parameter('pixel_query_topic', '/detector/pixel_query')
        self.declare_parameter('pixel_query_result_topic', '/detector/pixel_query_result_json')
        self.declare_parameter('robot_status_topic', '/robot/inspection_status')
        self.declare_parameter('depth_scale', 0.001)
        self.declare_parameter('depth_preview_min_m', 0.10)
        self.declare_parameter('depth_preview_max_m', 1.0)

    def _image_callback(self, msg):
        try:
            image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as exc:
            self.get_logger().warn('Failed to convert dashboard image: %s' % exc)
            return
        self.ingest_bgr_image(image, msg.header)

    def ingest_bgr_image(self, image, header):
        """Encode an in-memory BGR preview without publishing a ROS Image."""
        try:
            ok, encoded = cv2.imencode('.jpg', image, [int(cv2.IMWRITE_JPEG_QUALITY), self.state.jpeg_quality])
        except Exception as exc:
            self.get_logger().warn('Failed to encode dashboard image: %s' % exc)
            return
        if not ok:
            self.get_logger().warn('Failed to encode dashboard image as JPEG.')
            return

        with self.state.frame_condition:
            self.state.jpeg = encoded.tobytes()
            self.state.image_header = header
            self.state.image_width = int(image.shape[1])
            self.state.image_height = int(image.shape[0])
            self.state.last_update_time = time.time()
            self.state.frame_condition.notify_all()

    def ingest_depth(self, depth, depth_scale=None):
        """Colorize and cache an in-memory depth frame for the side preview."""
        try:
            scale = float(self.state.depth_scale if depth_scale is None else depth_scale)
            colored, valid_ratio = colorize_depth_bgr(
                depth,
                depth_scale=scale,
                min_m=self.state.depth_min_m,
                max_m=self.state.depth_max_m,
            )
            ok, encoded = cv2.imencode(
                '.jpg',
                colored,
                [int(cv2.IMWRITE_JPEG_QUALITY), max(40, min(85, self.state.jpeg_quality))],
            )
        except Exception as exc:
            self.get_logger().warn('Failed to encode depth preview: %s' % exc)
            return
        if not ok:
            self.get_logger().warn('Failed to encode depth preview as JPEG.')
            return

        depth_arr = np.asarray(depth)
        with self.state.depth_condition:
            self.state.depth_jpeg = encoded.tobytes()
            self.state.depth_raw = np.ascontiguousarray(depth_arr)
            self.state.depth_width = int(depth_arr.shape[1])
            self.state.depth_height = int(depth_arr.shape[0])
            self.state.depth_scale = scale
            self.state.depth_valid_ratio = float(valid_ratio)
            self.state.last_update_time = time.time()
            self.state.depth_condition.notify_all()

    def _objects_callback(self, msg):
        with self.state.lock:
            self.state.objects2d = msg
            self.state.last_update_time = time.time()

    def _objects3d_callback(self, msg):
        with self.state.lock:
            self.state.objects3d = msg
            self.state.last_update_time = time.time()

    def _target_point_callback(self, msg):
        with self.state.lock:
            self.state.target_point = msg
            self.state.last_update_time = time.time()

    def _target_pose_callback(self, msg):
        with self.state.lock:
            self.state.target_pose = msg
            self.state.last_update_time = time.time()

    def _joint_state_callback(self, msg):
        with self.state.lock:
            self.state.target_joint_state = msg
            self.state.last_update_time = time.time()

    def _current_joint_state_callback(self, msg):
        with self.state.lock:
            self.state.current_joint_state = msg
            self.state.last_update_time = time.time()

    def _current_ee_point_callback(self, msg):
        with self.state.lock:
            self.state.current_ee_point = msg
            self.state.last_update_time = time.time()

    def _objects_ik_callback(self, msg):
        try:
            payload = json.loads(msg.data) if msg.data else None
        except Exception as exc:
            self.get_logger().warn('Failed to parse objects IK JSON: %s' % exc)
            return
        with self.state.lock:
            self.state.objects_ik_json = payload
            self.state.last_update_time = time.time()

    def _pixel_query_result_callback(self, msg):
        try:
            payload = json.loads(msg.data) if msg.data else None
        except Exception as exc:
            self.get_logger().warn('Failed to parse pixel query result JSON: %s' % exc)
            return
        with self.state.pixel_query_condition:
            self.state.pixel_query_result = payload
            self.state.last_update_time = time.time()
            self.state.pixel_query_condition.notify_all()

    def _robot_status_callback(self, msg):
        with self.state.lock:
            self.state.robot_status = msg
            self.state.last_update_time = time.time()

    def _make_handler(self):
        state = self.state
        frame_timeout_sec = self.frame_timeout_sec
        pixel_query_pub = self.pixel_query_pub

        class DashboardHandler(BaseHTTPRequestHandler):
            protocol_version = 'HTTP/1.1'

            def do_GET(self):
                if self.path in ('/', '/index.html'):
                    self._send_bytes(INDEX_HTML.encode('utf-8'), 'text/html; charset=utf-8')
                    return
                if self.path.startswith('/api/state'):
                    self._send_json(self._snapshot())
                    return
                if self.path.startswith('/api/depth_sample'):
                    self._send_json(self._depth_sample())
                    return
                if self.path.startswith('/stream.mjpg'):
                    self._stream_mjpeg(kind='rgb')
                    return
                if self.path.startswith('/depth.mjpg'):
                    self._stream_mjpeg(kind='depth')
                    return
                self.send_error(404, 'not found')

            def do_POST(self):
                if self.path.startswith('/api/pick3d'):
                    self._handle_pick3d()
                    return
                self.send_error(404, 'not found')

            def log_message(self, fmt, *args):
                return

            def _send_bytes(self, payload, content_type):
                self.send_response(200)
                self.send_header('Content-Type', content_type)
                self.send_header('Content-Length', str(len(payload)))
                self.send_header('Cache-Control', 'no-store')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(payload)

            def _send_json(self, payload):
                data = json.dumps(payload, ensure_ascii=False).encode('utf-8')
                self._send_bytes(data, 'application/json; charset=utf-8')

            def _handle_pick3d(self):
                try:
                    length = min(65536, max(0, int(self.headers.get('Content-Length', '0'))))
                    payload = json.loads(self.rfile.read(length).decode('utf-8'))
                    u = float(payload['u'])
                    v = float(payload['v'])
                except Exception as exc:
                    self._send_json({'accepted': False, 'message': 'invalid request: %s' % exc})
                    return

                with state.lock:
                    header = state.image_header
                    width = int(state.image_width)
                    height = int(state.image_height)
                    if header is None or width <= 0 or height <= 0:
                        self._send_json({'accepted': False, 'message': 'no dashboard image available'})
                        return
                    u = max(0.0, min(float(width - 1), u))
                    v = max(0.0, min(float(height - 1), v))
                    state.pixel_query_sequence += 1
                    request_id = int(state.pixel_query_sequence)
                    stamp_sec = int(header.stamp.sec)
                    stamp_nanosec = int(header.stamp.nanosec)
                    frame_id = str(header.frame_id)

                query = PointStamped()
                query.header.stamp.sec = stamp_sec
                query.header.stamp.nanosec = stamp_nanosec
                query.header.frame_id = frame_id
                query.point.x = u
                query.point.y = v
                query.point.z = float(request_id)
                pixel_query_pub.publish(query)

                deadline = time.monotonic() + 3.0
                result = None
                with state.pixel_query_condition:
                    while time.monotonic() < deadline:
                        candidate = state.pixel_query_result
                        if isinstance(candidate, dict) and int(candidate.get('request_id', -1)) == request_id:
                            result = dict(candidate)
                            break
                        state.pixel_query_condition.wait(timeout=max(0.0, deadline - time.monotonic()))

                if result is None:
                    self._send_json(
                        {
                            'accepted': True,
                            'request_id': request_id,
                            'valid': False,
                            'pending': True,
                            'pixel': [u, v],
                            'message': 'query accepted; result pending',
                        }
                    )
                    return
                result['accepted'] = True
                self._send_json(result)

            def _stream_mjpeg(self, kind='rgb'):
                self.send_response(200)
                self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
                self.send_header('Cache-Control', 'no-store')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()

                last_frame = None
                condition = state.frame_condition if kind == 'rgb' else state.depth_condition
                while True:
                    with condition:
                        condition.wait(timeout=frame_timeout_sec)
                        frame = state.jpeg if kind == 'rgb' else state.depth_jpeg
                    if frame is None or frame == last_frame:
                        continue
                    last_frame = frame
                    try:
                        self.wfile.write(b'--frame\r\n')
                        self.wfile.write(b'Content-Type: image/jpeg\r\n')
                        self.wfile.write(('Content-Length: %d\r\n\r\n' % len(frame)).encode('ascii'))
                        self.wfile.write(frame)
                        self.wfile.write(b'\r\n')
                    except (BrokenPipeError, ConnectionResetError):
                        return

            def _depth_sample(self):
                from urllib.parse import parse_qs, urlparse

                query = parse_qs(urlparse(self.path).query)
                try:
                    nx = float((query.get('nx') or query.get('x') or ['0'])[0])
                    ny = float((query.get('ny') or query.get('y') or ['0'])[0])
                except (TypeError, ValueError):
                    return {'ok': False, 'message': 'invalid nx/ny'}
                nx = max(0.0, min(1.0, nx))
                ny = max(0.0, min(1.0, ny))
                with state.lock:
                    depth = state.depth_raw
                    width = int(state.depth_width)
                    height = int(state.depth_height)
                    scale = float(state.depth_scale)
                    if depth is None or width <= 0 or height <= 0:
                        return {'ok': False, 'message': 'no depth frame yet'}
                    u = int(round(nx * float(width - 1)))
                    v = int(round(ny * float(height - 1)))
                    u = max(0, min(width - 1, u))
                    v = max(0, min(height - 1, v))
                    raw = depth[v, u]
                    if np.issubdtype(np.asarray(raw).dtype, np.floating):
                        depth_m = float(raw)
                        raw_mm = int(round(depth_m / max(scale, 1e-9)))
                    else:
                        raw_mm = int(raw)
                        depth_m = float(raw_mm) * scale
                    valid = bool(np.isfinite(depth_m) and depth_m > 1e-6)
                    return {
                        'ok': True,
                        'u': u,
                        'v': v,
                        'nx': nx,
                        'ny': ny,
                        'raw_mm': raw_mm,
                        'depth_m': depth_m if valid else 0.0,
                        'valid': valid,
                        'depth_scale': scale,
                        'width': width,
                        'height': height,
                        'valid_ratio': float(state.depth_valid_ratio),
                    }

            def _snapshot(self):
                with state.lock:
                    return {
                        'server_time': time.time(),
                        'last_update_time': state.last_update_time,
                        'diagnostics': dashboard_diagnostics(state, time.time()),
                        'image': {
                            'header': header_to_dict(state.image_header),
                            'width': state.image_width,
                            'height': state.image_height,
                            'jpeg_quality': state.jpeg_quality,
                        },
                        'depth': {
                            'width': state.depth_width,
                            'height': state.depth_height,
                            'depth_scale': state.depth_scale,
                            'valid_ratio': state.depth_valid_ratio,
                            'min_m': state.depth_min_m,
                            'max_m': state.depth_max_m,
                        },
                        'objects': dashboard_objects_to_dict(state.objects3d, state.objects_ik_json),
                        'objects2d': object2d_array_to_dict(state.objects2d),
                        'objects3d': object3d_array_to_dict(state.objects3d),
                        'objects_ik': state.objects_ik_json or {'objects': [], 'message': 'no data'},
                        'robot_status': robot_status_to_dict(state.robot_status),
                        'target_point': point_stamped_to_dict(state.target_point),
                        'target_pose': pose_stamped_to_dict(state.target_pose),
                        'target_joint_state': joint_state_to_dict(state.target_joint_state),
                        'current_joint_state': joint_state_to_dict(state.current_joint_state),
                        'current_ee_point': point_stamped_to_dict(state.current_ee_point),
                        'pixel_query_result': state.pixel_query_result,
                    }

        return DashboardHandler

    def destroy_node(self):
        try:
            self.server.shutdown()
            self.server.server_close()
        finally:
            super().destroy_node()


def stamp_to_dict(stamp):
    if stamp is None:
        return {'sec': 0, 'nanosec': 0}
    return {'sec': int(stamp.sec), 'nanosec': int(stamp.nanosec)}


def header_to_dict(header):
    if header is None:
        return None
    return {'stamp': stamp_to_dict(header.stamp), 'frame_id': str(header.frame_id)}


def point_to_dict(point):
    if point is None:
        return {'x': 0.0, 'y': 0.0, 'z': 0.0}
    return {'x': float(point.x), 'y': float(point.y), 'z': float(point.z)}


def quaternion_to_dict(quat):
    if quat is None:
        return {'x': 0.0, 'y': 0.0, 'z': 0.0, 'w': 1.0}
    return {'x': float(quat.x), 'y': float(quat.y), 'z': float(quat.z), 'w': float(quat.w)}


def object2d_to_dict(obj):
    payload = {
        'class_name': str(obj.class_name),
        'class_id': int(obj.class_id),
        'confidence': float(obj.confidence),
        'xmin': int(obj.xmin),
        'ymin': int(obj.ymin),
        'xmax': int(obj.xmax),
        'ymax': int(obj.ymax),
        'cx': float(obj.cx),
        'cy': float(obj.cy),
    }
    edge_px = [float(v) for v in getattr(obj, 'handle_grasp_edge_px', [])]
    center_px = [float(v) for v in getattr(obj, 'handle_grasp_center_px', [])]
    if len(edge_px) == 4:
        payload['handle_grasp_edge_px'] = [[edge_px[0], edge_px[1]], [edge_px[2], edge_px[3]]]
    if len(center_px) == 2:
        payload['handle_grasp_center_px'] = [center_px[0], center_px[1]]
    width_px = float(getattr(obj, 'handle_grasp_width_px', 0.0))
    if width_px > 0.0:
        payload['handle_grasp_width_px'] = width_px
    source = str(getattr(obj, 'handle_grasp_source', ''))
    if source:
        payload['handle_grasp_source'] = source
    payload['label_text'] = str(getattr(obj, 'label_text', ''))
    payload['label_confidence'] = float(getattr(obj, 'label_confidence', 0.0))
    payload['semantic_name'] = str(getattr(obj, 'semantic_name', ''))
    payload['control_id'] = str(getattr(obj, 'control_id', ''))
    payload['spatial_relation'] = str(getattr(obj, 'spatial_relation', ''))
    payload['label_tag_present'] = bool(getattr(obj, 'label_tag_present', False))
    geom = [float(v) for v in getattr(obj, 'geometric_center_px', [])]
    if len(geom) >= 2:
        payload['geometric_center_px'] = [geom[0], geom[1]]
    return payload


def object2d_array_to_dict(msg):
    if msg is None:
        return {'header': None, 'objects': []}
    return {
        'header': header_to_dict(msg.header),
        'objects': [object2d_to_dict(obj) for obj in msg.objects],
    }


def object3d_to_dict(obj):
    return {
        'header': header_to_dict(obj.header),
        'detection': object2d_to_dict(obj.detection),
        'valid': bool(obj.valid),
        'cached': bool(obj.cached),
        'source_frame': str(obj.source_frame),
        'target_frame': str(obj.target_frame),
        'depth_m': float(obj.depth_m),
        'point_camera': point_to_dict(obj.point_camera),
        'point_target': point_to_dict(obj.point_target),
        'message': str(obj.message),
    }


def object3d_array_to_dict(msg):
    if msg is None:
        return {'header': None, 'objects': []}
    return {
        'header': header_to_dict(msg.header),
        'objects': [object3d_to_dict(obj) for obj in msg.objects],
    }


def dashboard_diagnostics(state, now):
    warnings = []
    if state.image_header is None:
        warnings.append(
            'WARN: no /detector/debug_image received yet. '
            'Check /camera/color/image_raw, yolo_detector_node, and detector backend logs.'
        )
    if state.objects2d is None:
        warnings.append('WARN: no /detector/objects_2d received yet.')
    if state.objects3d is None:
        warnings.append('WARN: no /detector/objects received yet. Check depth image and CameraInfo.')
    if state.current_joint_state is None:
        warnings.append('WARN: no /detector/current_joint_state received yet. Check Unitree lowstate topic/interface.')
    if state.last_update_time <= 0.0:
        warnings.append('WARN: dashboard has not received any subscribed ROS message.')
    else:
        age = float(now) - float(state.last_update_time)
        if age > 5.0:
            warnings.append('WARN: last dashboard update is %.1fs old; ROS topics may be stalled.' % age)
    return warnings


def dashboard_objects_to_dict(objects3d_msg, objects_ik_json):
    if objects3d_msg is None:
        return []

    ik_by_id = {}
    if isinstance(objects_ik_json, dict):
        for item in objects_ik_json.get('objects', []):
            object_id = item.get('object_id')
            if object_id:
                ik_by_id[object_id] = item

    objects = []
    for index, obj in enumerate(objects3d_msg.objects):
        det = obj.detection
        object_id = object_id_from_detection(index, det.class_name)
        ik_item = ik_by_id.get(object_id, {})
        target_point = point_to_dict(obj.point_target)
        camera_point = point_to_dict(obj.point_camera)
        item = {
            'object_id': object_id,
            'class_name': str(det.class_name),
            'label_text': str(getattr(det, 'label_text', '')),
            'label_confidence': float(getattr(det, 'label_confidence', 0.0)),
            'semantic_name': str(getattr(det, 'semantic_name', '')),
            'control_id': str(getattr(det, 'control_id', '')),
            'spatial_relation': str(getattr(det, 'spatial_relation', '')),
            'label_tag_present': bool(getattr(det, 'label_tag_present', False)),
            'class_id': int(det.class_id),
            'confidence': float(det.confidence),
            'bbox_xyxy': [int(det.xmin), int(det.ymin), int(det.xmax), int(det.ymax)],
            'center_px': [float(det.cx), float(det.cy)],
            'valid_3d': bool(obj.valid),
            'cached': bool(obj.cached),
            'source_frame': str(obj.source_frame),
            'target_frame': str(obj.target_frame),
            'depth_m': float(obj.depth_m),
            'point_camera': camera_point,
            'point_target': target_point,
            'point_cam_m': [camera_point['x'], camera_point['y'], camera_point['z']],
            'point_torso_m': [target_point['x'], target_point['y'], target_point['z']] if obj.valid else None,
            'message': str(obj.message),
            'ik': ik_item.get('ik'),
        }
        for key in (
            'handle_grasp_edge_px',
            'handle_grasp_center_px',
            'handle_mid_px',
            'handle_grasp_width_px',
            'handle_grasp_source',
            'handle_grasp_endpoints_world_m',
            'handle_grasp_endpoint_targets_m',
            'handle_grasp_center_world_m',
            'handle_grasp_ree_target_m',
            'handle_mid_surface_world_m',
            'handle_mid_right_air_world_m',
            'handle_mid_right_air_target_m',
            'handle_mid_right_offset_m',
            'handle_mid_right_air_detail',
            'handle_grasp_width_m',
            'handle_grasp_endpoint_details',
            'handle_grasp_message',
            'handle_depth_near_cut_m',
            'handle_depth_background_m',
            'handle_depth_mask_area_px',
            'handle_grasp_long_axis_length_px',
            'handle_depth_tip_distance_px',
            'geometric_center_px',
            'detection_source',
            'seg_geometric_center_world_m',
            'seg_geometric_center_target_m',
            'seg_geometric_center_detail',
            'seg_geometry_message',
        ):
            if key in ik_item:
                item[key] = ik_item[key]
        objects.append(item)
    return objects


def object_id_from_detection(index, class_name):
    safe = ''.join(ch if ch.isalnum() else '_' for ch in str(class_name).lower()).strip('_')
    return '%02d_%s' % (int(index), safe or 'object')


def point_stamped_to_dict(msg):
    if msg is None:
        return {'header': None, 'point': point_to_dict(None)}
    return {'header': header_to_dict(msg.header), 'point': point_to_dict(msg.point)}


def pose_stamped_to_dict(msg):
    if msg is None:
        return {
            'header': None,
            'pose': {
                'position': point_to_dict(None),
                'orientation': quaternion_to_dict(None),
            },
        }
    return {
        'header': header_to_dict(msg.header),
        'pose': {
            'position': point_to_dict(msg.pose.position),
            'orientation': quaternion_to_dict(msg.pose.orientation),
        },
    }


def joint_state_to_dict(msg):
    if msg is None:
        return {'header': None, 'name': [], 'position': [], 'velocity': [], 'effort': []}
    return {
        'header': header_to_dict(msg.header),
        'name': [str(name) for name in msg.name],
        'position': [float(value) for value in msg.position],
        'velocity': [float(value) for value in msg.velocity],
        'effort': [float(value) for value in msg.effort],
    }


def robot_status_to_dict(msg):
    if msg is None:
        return {
            'available': False,
            'header': None,
            'stage_id': 0,
            'stage_name': '',
            'current_action': '',
            'motion_active': False,
            'progress': 0.0,
            'has_error': False,
            'error_code': '',
            'error_message': '',
            'emergency_stop': False,
            'target_reachable': False,
            'reachability_message': 'no /robot/inspection_status received',
            'target_id': '',
        }
    return {
        'available': True,
        'header': header_to_dict(msg.header),
        'stage_id': int(msg.stage_id),
        'stage_name': str(msg.stage_name),
        'current_action': str(msg.current_action),
        'motion_active': bool(msg.motion_active),
        'progress': float(msg.progress),
        'has_error': bool(msg.has_error),
        'error_code': str(msg.error_code),
        'error_message': str(msg.error_message),
        'emergency_stop': bool(msg.emergency_stop),
        'target_reachable': bool(msg.target_reachable),
        'reachability_message': str(msg.reachability_message),
        'target_id': str(msg.target_id),
    }


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = WebDashboardNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
