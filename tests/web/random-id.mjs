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
