import React, { useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { returnToWorkspace } from "./return-to-workspace";

export default function SubmissionComplete({ completion, onKeepWorking }) {
  const dialog = useRef(null);
  const returnButton = useRef(null);
  const [returning, setReturning] = useState(false);
  const annotation = completion.kind === "annotation";
  const needsChanges = completion.kind === "changes_requested";
  useEffect(() => {
    const element = dialog.current;
    element.showModal();
    returnButton.current.focus();
    return () => element.close();
  }, []);

  return createPortal(
    <dialog
      ref={dialog}
      className="monailabel-complete"
      aria-labelledby="submission-title"
      aria-describedby="submission-description"
      onKeyDownCapture={(event) => {
        // Keep OHIF's global editing shortcuts out of the completion dialog.
        event.stopPropagation();
        if (event.key === "Escape") {
          event.preventDefault();
          if (!returning) onKeepWorking();
        }
      }}
      onCancel={(event) => {
        event.preventDefault();
        if (!returning) onKeepWorking();
      }}
    >
      <svg
        className="monailabel-complete-icon"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        strokeWidth="1.7"
        aria-hidden="true"
      >
        <circle cx="12" cy="12" r="10" />
        <path d="m7 12 3 3 7-7" />
      </svg>
      <h2 id="submission-title">
        {annotation ? "Annotation submitted" : "Review submitted"}
      </h2>
      <p className="monailabel-complete-sample">
        {completion.name} · Revision {completion.revision}
      </p>
      <div id="submission-description">
        <p>
          {annotation
            ? "Your annotation is saved and waiting for review. You can move on to another sample."
            : needsChanges
              ? "Your review is saved as Needs changes. This sample has been returned for correction."
              : "Your review is saved as Good, including any segmentation corrections."}
        </p>
        <p>
          {annotation
            ? "Return to Datasets to choose your next sample."
            : "Return to Reviews to choose the next pending case."}
        </p>
        {(needsChanges || completion.hasLocalEdits) && (
          <p className="monailabel-complete-note">
            {needsChanges
              ? "Local segmentation edits are not included in a Needs changes decision."
              : "Edits made while submitting are still local. Keep working to submit them."}
          </p>
        )}
      </div>
      <div className="monailabel-complete-actions">
        <button disabled={returning} onClick={onKeepWorking}>
          Keep working
        </button>
        <button
          ref={returnButton}
          className="monailabel-complete-primary"
          disabled={returning}
          onClick={async () => {
            setReturning(true);
            await returnToWorkspace(
              completion.projectId,
              annotation ? "datasets" : "review",
            );
          }}
        >
          {returning ? "Returning…" : "Return to workspace"}
        </button>
      </div>
    </dialog>,
    document.body,
  );
}
