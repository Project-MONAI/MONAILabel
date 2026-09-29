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

import assert from "node:assert/strict";
import { webcrypto } from "node:crypto";
import { readFileSync } from "node:fs";
import test from "node:test";
import { runInNewContext } from "node:vm";

for (const [client, path] of [
  [
    "workspace",
    "../../packages/server/src/monailabel/server/static/random-id.js",
  ],
  [
    "OHIF",
    "../../packages/viewers/src/monailabel/viewers/resources/ohif/extension/src/random-id.js",
  ],
])
  test(`${client} IDs work without the HTTPS-only randomUUID API`, () => {
    const source = readFileSync(new URL(path, import.meta.url), "utf8");
    const randomId = runInNewContext(
      source.replace("export function", "function") + "; randomId;",
      {
        crypto: { getRandomValues: webcrypto.getRandomValues.bind(webcrypto) },
      },
    );
    const identifiers = Array.from({ length: 100 }, randomId);
    assert.equal(new Set(identifiers).size, identifiers.length);
    for (const identifier of identifiers)
      assert.match(identifier, /^[0-9a-f]{32}$/);
  });
