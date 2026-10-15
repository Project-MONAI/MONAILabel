/*
Copyright (c) MONAI Consortium
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

import React, { useEffect, useRef, useState } from "react";

export default function DemoCountdown() {
  const banner = useRef<HTMLParagraphElement>(null);
  const [expired, setExpired] = useState(false);
  const [unavailable, setUnavailable] = useState(false);
  useEffect(() => {
    let disposed = false;
    let clock;
    // Same-origin browser contract: reuse the workspace countdown without
    // importing server code into the viewer package or starting another lease.
    const sessionModule = "/static/demo-session.js";
    import(/* webpackIgnore: true */ sessionModule)
      .then(({ demoSession }) => {
        if (disposed) return;
        clock = demoSession({
          banner: banner.current,
          trackActivity: false,
          // Keep the local draft in memory. Protected APIs enforce expiry;
          // navigating away here would discard edits without an explicit action.
          onExpired: () => setExpired(true),
        });
        return clock.start();
      })
      .catch(() => { if (!disposed) setUnavailable(true); });
    return () => {
      disposed = true;
      clock?.dispose();
    };
  }, []);
  return (
    <div className="monailabel-demo-session">
      <p ref={banner} hidden role="timer" aria-label="Demo time remaining" aria-live="off" />
      {unavailable && <p role="alert">Unable to load the demo clock. Check your connection to the workspace.</p>}
      {expired && <a href="/">Return to the workspace</a>}
    </div>
  );
}
