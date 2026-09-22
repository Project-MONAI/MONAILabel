import React, { useEffect, useRef, useState } from "react";
import { createSpeech } from "./speech";
import { panelSession, usePanelState } from "./panel-session";
import { api } from "./api";
import { randomId } from "./random-id";
import "./panel.css";
import {
  drawHint,
  spatialObjects,
  applySpatial,
  applyRegion,
  checkSpatial,
} from "./spatial-hints";
import { mergeMask } from "./masks";
import { clearMask } from "./segmentation-edits";
import { initializeAnnotation } from "./initialize";
import SubmissionComplete from "./SubmissionComplete";

export default function Panel({ servicesManager }) {
  const assetId = new URLSearchParams(location.search).get("assetId");
  const reviewMode =
    new URLSearchParams(location.search).get("mode") === "review";
  const session = panelSession(servicesManager, assetId);
  const [messages, setMessages] = usePanelState(session, "messages", []);
  const [prompt, setPrompt] = usePanelState(session, "prompt", "");
  const [models] = usePanelState(session, "models", []);
  const [selected, setSelected] = usePanelState(session, "selected", "");
  const [busy, setBusy] = usePanelState(session, "busy", false);
  const [jobStatus, setJobStatus] = usePanelState(session, "jobStatus", "");
  const [ready] = usePanelState(session, "ready", false);
  const [loadError] = usePanelState(session, "loadError", "");
  const [loadingText] = usePanelState(session, "loadingText", "Opening image…");
  const [completion, setCompletion] = usePanelState(
    session,
    "completion",
    null,
  );
  const [permission] = usePanelState(session, "permission", false);
  const [canReview] = usePanelState(session, "canReview", false);
  const [reviewComment, setReviewComment] = usePanelState(
    session,
    "reviewComment",
    "",
  );
  const [voiceState, setVoiceState] = useState("idle");
  const [voiceNote, setVoiceNote] = useState("");
  const [voiceAvailable, setVoiceAvailable] = useState(false);
  const [readReplies, setReadReplies] = usePanelState(
    session,
    "readReplies",
    false,
  );
  const voice = useRef(null);
  const spokenReplies = useRef(false);
  const promptValue = useRef("");
  promptValue.current = prompt;
  spokenReplies.current = readReplies;
  useEffect(() => {
    const speech = createSpeech({
      getText: () => promptValue.current,
      onText: setPrompt,
      onState: setVoiceState,
      onError: setVoiceNote,
    });
    voice.current = speech;
    setVoiceAvailable(speech.available);
    setVoiceNote(speech.unavailable);
    return () => {
      speech.dispose();
      voice.current = null;
    };
  }, []);
  const messagesEnd = useRef(null);
  useEffect(() => {
    messagesEnd.current?.scrollIntoView({ block: "nearest" });
  }, [messages]);
  const state = session.state;
  const log = (text) => {
    setMessages((items) => [...items, text]);
    if (spokenReplies.current && !text.startsWith("You: "))
      voice.current?.speak(text);
  };
  const run = async (task) => {
    if (busy) return;
    setBusy(true);
    try {
      await task();
    } catch (error) {
      log(error.message);
    } finally {
      setBusy(false);
    }
  };
  const initialize = async () => {
    if (!(await initializeAnnotation(state.current, api)))
      throw new Error("Wait for the source image to finish loading.");
  };
  const remember = (mask) => {
    const s = state.current;
    s.undo.push(mask);
    s.redo = [];
    while (
      s.undo.length > 5 ||
      s.undo.reduce((sum, item) => sum + item.length, 0) > 128 * 1024 ** 2
    )
      s.undo.shift();
  };
  const send = () =>
    run(async () => {
      voice.current?.cancel();
      const text = prompt.trim();
      if (!text) return;
      setPrompt("");
      log("You: " + text);
      const s = state.current;
      if (!permission && !canReview)
        throw new Error("Project permission is required.");
      await initialize();
      const before = s.transfer.read();
      // read() returns a detached source-grid snapshot. Exact comparison also
      // works on network HTTP origins, where Web Crypto's digest is unavailable.
      const checkDraft = (current, message) => {
        if (
          before.length !== current.length ||
          before.some((value, index) => value !== current[index])
        )
          throw new Error(message);
      };
      const context = {
        asset_id: s.asset.id,
        base_revision: s.asset.revision,
        slice: s.transfer.scope(),
        viewer_actions: [
          "box",
          "edit_spatial_prompts",
          "clear_segments",
          "undo",
          "redo",
          ...(!reviewMode && permission ? ["submit"] : []),
          ...(canReview ? ["review_annotation"] : []),
        ],
      };
      context.spatial_objects = spatialObjects(
        servicesManager.services,
        s.asset,
      );
      if (selected) context.model_id = selected;
      const reply = await api("/projects/" + s.project.id + "/assistant", {
        message: text,
        context,
        conversation_id: s.conversationId || null,
        request_id: randomId(),
      });
      s.conversationId = reply.conversation_id;
      log(reply.message);
      if (reply.data?.model_id) setSelected(reply.data.model_id);
      if (s.closed || state.current !== s)
        throw new Error("The viewer session changed. Retry the prompt.");
      if (reply.data?.client_action === "edit_spatial_prompts") {
        applySpatial(
          servicesManager.services,
          s.asset,
          s.project.id,
          reply.data,
          s.transfer.scope(),
        );
        log("SAM hints updated. Drag them to refine; segmentation unchanged.");
        return;
      }
      if (reply.data?.client_action === "viewer_edit") {
        const edit = reply.data;
        if (
          edit.asset_id !== s.asset.id ||
          edit.project_id !== s.project.id ||
          edit.base_revision !== s.asset.revision
        )
          throw new Error(
            "Viewer action belongs to another sample or revision.",
          );
        checkDraft(
          s.transfer.read(),
          "Your mask changed while interpreting the prompt. Retry it.",
        );
        if (edit.operation === "submit") {
          await submitAnnotation();
          return;
        }
        if (!["undo", "redo"].includes(edit.operation))
          throw new Error("Unsupported viewer action.");
        const from = edit.operation === "undo" ? s.undo : s.redo;
        const to = edit.operation === "undo" ? s.redo : s.undo;
        if (!from.length)
          throw new Error("No assistant change to " + edit.operation + ".");
        to.push(s.transfer.read());
        s.transfer.write(from.pop());
        log("Restored the previous mask.");
        return;
      }
      if (reply.data?.client_action === "clear_segments") {
        const action = reply.data;
        if (
          action.project_id !== s.project.id ||
          action.asset_id !== s.asset.id ||
          action.base_revision !== s.asset.revision ||
          action.label_ids.some(
            (id) => !s.project.labels.some((label) => label.id === id),
          )
        )
          throw new Error(
            "Clear request belongs to another sample, revision or label.",
          );
        if (
          action.slice &&
          (action.slice.axis !== context.slice.axis ||
            action.slice.index !== context.slice.index)
        )
          throw new Error(
            "Clear request differs from the requested source slice.",
          );
        checkDraft(
          s.transfer.read(),
          "Your mask changed while requesting the edit. Retry it.",
        );
        const cleared = clearMask(before, action, s.asset.spatial_shape);
        if (cleared.every((value, index) => value === before[index])) {
          log("No matching annotations to clear.");
          return;
        }
        remember(before);
        s.transfer.write(cleared);
        log("Cleared the requested annotation. Say ‘undo’ to restore it.");
        return;
      }
      if (reply.data?.project?.id === s.project.id) {
        s.project = reply.data.project;
        s.transfer.syncProject(s.project);
      }
      if (reply.data?.client_action === "review_annotation") {
        const action = reply.data;
        if (
          action.asset_id !== s.asset.id ||
          action.base_revision !== s.asset.revision
        )
          throw new Error(
            "Review action belongs to another sample or revision.",
          );
        checkDraft(
          s.transfer.read(),
          "Your mask changed while interpreting the review. Retry it.",
        );
        await reviewDecision(action.verdict, action.comment || "");
        return;
      }
      if (!reply.job_id) return;
      s.job = reply.job_id;
      let job;
      try {
        do {
          await new Promise((resolve) => setTimeout(resolve, 700));
          job = await api("/jobs/" + s.job);
          setJobStatus(job.progress_message || job.status);
        } while (["queued", "running"].includes(job.status));
      } finally {
        s.job = null;
        setJobStatus("");
      }
      if (job.status !== "succeeded") throw new Error(job.error || job.status);
      if (s.closed || state.current !== s)
        throw new Error("The viewer session changed.");
      if (job.result.region_id) {
        const region = await api("/regions/" + job.result.region_id);
        const added = applyRegion(
          servicesManager.services,
          s.asset,
          s.project.id,
          region,
          context.spatial_objects,
          context.slice,
          s.transfer.scope(),
        );
        log(
          added
            ? "Added an editable box. Drag its handles to adjust it."
            : "No target found; no box added.",
        );
        return;
      }
      if (!job.result.proposal_id) {
        log("Job completed.");
        return;
      }
      const proposal = await api("/proposals/" + job.result.proposal_id);
      const mask = await api(
        "/proposals/" + proposal.id + "/mask.bin",
        undefined,
        true,
      );
      if (proposal.spatial_prompt)
        checkSpatial(
          servicesManager.services,
          s.asset,
          context.spatial_objects,
          context.slice,
          s.transfer.scope(),
        );
      const current = s.transfer.read();
      checkDraft(
        current,
        "Local edits changed during inference. The proposal was not applied.",
      );
      if (
        proposal.asset_id !== s.asset.id ||
        proposal.base_revision !== s.asset.revision
      )
        throw new Error("Proposal belongs to another sample or revision.");
      const merged = mergeMask(current, mask, proposal, s.asset.spatial_shape);
      remember(current);
      s.transfer.write(merged);
      s.proposal = proposal.id;
      log(
        "Applied the editable proposal. Inspect every required structure before submitting for review.",
      );
    });
  const submitAnnotation = async () => {
    if (reviewMode)
      throw new Error(
        "Use the review actions to submit your decision and corrections.",
      );
    if (!permission) throw new Error("Annotator permission is required.");
    await initialize();
    const s = state.current;
    const query = new URLSearchParams({ base_revision: s.asset.revision });
    s.project.labels.forEach((label) =>
      query.append("covered_labels", label.id),
    );
    if (s.proposal) query.set("proposal_id", s.proposal);
    const mask = s.transfer.read();
    const annotation = await api(
      "/assets/" + s.asset.id + "/review-mask?" + query,
      mask,
    );
    s.asset = {
      ...s.asset,
      revision: annotation.revision,
      annotation_id: annotation.id,
    };
    s.proposal = null;
    log(
      "Revision " +
        annotation.revision +
        " saved and waiting for review. Return to Datasets to choose your next sample, or keep working here.",
    );
    setCompletion({
      kind: "annotation",
      name: s.asset.name,
      revision: annotation.revision,
      projectId: s.asset.project_id,
      hasLocalEdits: s.transfer
        .read()
        .some((value, index) => value !== mask[index]),
    });
  };
  const submit = () => run(submitAnnotation);
  const reviewDecision = async (verdict, comment = reviewComment) => {
    const s = state.current;
    if (!canReview || !s.asset?.annotation_id)
      throw new Error("Open a submitted annotation with reviewer permission.");
    await initialize();
    let hasLocalEdits = false;
    if (verdict === "accepted") {
      const metadata = JSON.stringify({
        base_revision: s.asset.revision,
        covered_labels: s.project.labels.map((l) => l.id),
        comment,
      }).replace(
        /[\u007f-\uffff]/g,
        (ch) => "\\u" + ch.charCodeAt(0).toString(16).padStart(4, "0"),
      );
      const mask = s.transfer.read();
      const annotation = await api(
        "/assets/" + s.asset.id + "/review-complete",
        mask,
        false,
        { "X-MONAILABEL-REVIEW": metadata },
      );
      s.asset = {
        ...s.asset,
        revision: annotation.revision,
        annotation_id: annotation.id,
      };
      s.proposal = null;
      hasLocalEdits = s.transfer
        .read()
        .some((value, index) => value !== mask[index]);
      log(
        "Review saved: Good · revision " +
          annotation.revision +
          ". Return to Reviews to choose your next pending case, or keep working here.",
      );
    } else {
      const decision = await api(
        "/annotations/" + s.asset.annotation_id + "/decision",
        { verdict: "changes_requested", comment },
      );
      log(
        "Review saved: Needs changes · revision " +
          decision.revision +
          ". Return to Reviews to choose your next pending case, or keep working here.",
      );
    }
    setReviewComment("");
    setCompletion({
      kind: verdict,
      name: s.asset.name,
      revision: s.asset.revision,
      projectId: s.asset.project_id,
      hasLocalEdits,
    });
  };
  return (
    <div className="monailabel-assistant">
      {completion && (
        <SubmissionComplete
          completion={completion}
          onKeepWorking={() => setCompletion(null)}
        />
      )}
      <div className="monailabel-panel-heading">
        <h3>{reviewMode ? "Review" : "Annotation"}</h3>
        <a href="/" target="_blank" rel="noreferrer">
          Workspace ↗
        </a>
      </div>
      <select
        aria-label="Annotation model"
        value={selected}
        onChange={(e) => setSelected(e.target.value)}
      >
        <option value="">Project defaults</option>
        {models.map((model) => (
          <option key={model.id} value={model.id}>
            {model.name}
          </option>
        ))}
      </select>
      {["sam2", "medsam2"].includes(
        models.find((m) => m.id === selected)?.provider,
      ) && (
        <details className="monailabel-review" open>
          <summary>Spatial prompt</summary>
          <p>
            Create a box or positive/negative points through chat using source
            voxel coordinates, then drag to refine. One target’s box and points
            are combined; select a box when several match.
          </p>
          <div style={{ display: "flex", gap: 6 }}>
            <button
              disabled={busy || !ready}
              onClick={() =>
                run(async () =>
                  drawHint(servicesManager.services, "RectangleROI"),
                )
              }
            >
              Box
            </button>
            <button
              disabled={busy || !ready}
              onClick={() =>
                run(async () => drawHint(servicesManager.services, "Probe"))
              }
            >
              Point
            </button>
          </div>
        </details>
      )}
      {!ready && (
        <div
          role={loadError ? "alert" : "status"}
          className="monailabel-loading"
        >
          {loadError ? (
            <>
              <span>Could not open the annotation: {loadError}</span>
              <button onClick={() => session.retry()}>Retry</button>
            </>
          ) : (
            loadingText
          )}
        </div>
      )}
      {canReview && (
        <details className="monailabel-review" open={reviewMode || undefined}>
          <summary>Review annotation</summary>
          <input
            aria-label="Review comment"
            placeholder="Review comment (optional)"
            value={reviewComment}
            onChange={(e) => setReviewComment(e.target.value)}
          />
          <div
            style={{ display: "flex", gap: 6, marginTop: 8, flexWrap: "wrap" }}
          >
            <button
              className="monailabel-button"
              disabled={!ready || busy || !state.current.asset?.annotation_id}
              onClick={() => run(() => reviewDecision("accepted"))}
            >
              Good
            </button>
            <button
              className="monailabel-button"
              disabled={!ready || busy || !state.current.asset?.annotation_id}
              onClick={() => run(() => reviewDecision("changes_requested"))}
            >
              Bad / needs changes
            </button>
            <button
              className="monailabel-button"
              disabled={!ready || busy || !state.current.asset?.annotation_id}
              onClick={() => run(() => reviewDecision("accepted"))}
            >
              Good with corrections
            </button>
          </div>
        </details>
      )}
      <div
        role="log"
        style={{
          overflowY: "auto",
          flex: 1,
          minHeight: 80,
          whiteSpace: "pre-wrap",
        }}
      >
        {messages.map((message, i) => (
          <p key={i} style={{ marginBottom: 10 }}>
            {message}
          </p>
        ))}
        <div ref={messagesEnd} />
      </div>
      <div className="monailabel-composer">
        <textarea
          aria-label="Annotation prompt"
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          onKeyDown={(e) => {
            if (e.ctrlKey && e.key === "Enter") {
              e.preventDefault();
              send();
            }
          }}
          placeholder="Ask about this image…"
          rows={3}
        />
        <div className="monailabel-composer-actions">
          <label>
            <input
              type="checkbox"
              checked={readReplies}
              disabled={!voice.current?.canSpeak}
              onChange={(event) => {
                setReadReplies(event.target.checked);
                if (event.target.checked)
                  voice.current?.speak("Spoken replies on.");
                else voice.current?.stopSpeaking();
              }}
            />
            Read replies
          </label>
          <div className="monailabel-composer-buttons">
            <button
              type="button"
              className="monailabel-microphone"
              aria-label={
                voiceState === "idle" ? "Use microphone" : "Stop microphone"
              }
              aria-pressed={voiceState !== "idle"}
              aria-describedby="monailabel-voice-status"
              disabled={!voiceAvailable || !ready || busy}
              title={
                voice.current?.unavailable ||
                (voiceState === "idle" ? "Use microphone" : "Stop microphone")
              }
              onClick={() => {
                setVoiceNote("");
                if (voiceState === "idle") voice.current?.start();
                else voice.current?.stop();
              }}
            >
              {voiceState === "idle" ? (
                <svg
                  viewBox="0 0 24 24"
                  width="20"
                  height="20"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="1.8"
                  strokeLinecap="round"
                  aria-hidden="true"
                >
                  <rect x="9" y="2" width="6" height="12" rx="3" />
                  <path d="M5 10v1a7 7 0 0 0 14 0v-1M12 18v4M8 22h8" />
                </svg>
              ) : (
                <svg
                  viewBox="0 0 24 24"
                  width="20"
                  height="20"
                  fill="currentColor"
                  aria-hidden="true"
                >
                  <rect x="5" y="5" width="14" height="14" rx="2" />
                </svg>
              )}
            </button>
            <button
              type="button"
              className="monailabel-send"
              aria-label="Send prompt"
              title="Send prompt"
              disabled={!ready || busy || (!permission && !canReview)}
              onClick={send}
            >
              <svg
                viewBox="0 0 24 24"
                width="20"
                height="20"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
                aria-hidden="true"
              >
                <path d="m6 10 6-6 6 6M12 4v16" />
              </svg>
            </button>
          </div>
        </div>
      </div>
      {jobStatus && <small role="status">{jobStatus}</small>}
      <small id="monailabel-voice-status" role="status">
        {voiceNote ||
          (voiceState === "starting"
            ? "Starting microphone…"
            : voiceState === "listening"
              ? "Listening… tap the stop button when finished."
              : "")}
      </small>
      {busy && (
        <button
          className="monailabel-button"
          onClick={() => {
            const job = state.current.job;
            if (job)
              api("/jobs/" + job + "/cancel", {}).catch((e) => log(e.message));
          }}
        >
          Cancel job
        </button>
      )}
      {!reviewMode && permission && (
        <button
          className="monailabel-button"
          disabled={!ready || busy || !permission}
          onClick={submit}
        >
          Submit for review
        </button>
      )}
    </div>
  );
}
