// Captures the README screenshots and browser timings from the running page.
// Skipped unless CAPTURE is set: it overwrites tracked files.
const fs = require("fs");
const path = require("path");
const { test } = require("@playwright/test");

const finished = () => window.loopclosure.finished === true;
const repo = path.join(__dirname, "..");

test.skip(!process.env.CAPTURE, "run with CAPTURE=1");

test("capture", async ({ page }) => {
  const timings = {};
  for (const [name, query, file] of [
    ["office", "input=office&seed=0&speed=1000", "demo_office.png"],
    ["warehouse", "input=warehouse&seed=0&speed=1000", "demo_warehouse.png"],
    ["intel", "input=intel&speed=1000", "demo_intel.png"],
  ]) {
    await page.goto(`index.html?${query}`);
    const started = Date.now();
    await page.waitForFunction(finished);
    const seconds = (Date.now() - started) / 1000;
    const state = await page.evaluate(() => window.loopclosure);
    timings[name] = {
      keyframes: state.keyframes,
      closures: state.closures,
      wall_seconds_with_drawing: seconds,
      recent_step_ms: state.stepMs,
    };
    await page.waitForTimeout(600);
    await page.screenshot({ path: path.join(repo, "docs", file), fullPage: true });
  }

  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("index.html?input=atrium&seed=0&speed=1000");
  await page.waitForFunction(finished);
  await page.waitForTimeout(300);
  await page.screenshot({ path: path.join(repo, "docs", "demo_phone.png"), fullPage: true });

  fs.writeFileSync(
    path.join(repo, "results", "browser.json"),
    `${JSON.stringify(
      {
        command: "cd e2e && npm run capture",
        browser: "Playwright Chromium, headless",
        hardware: "Apple M3 Max, 36 GB; other jobs were running on the machine",
        note: "each keyframe runs two SLAM instances, and wall time includes redrawing both maps",
        runs: timings,
      },
      null,
      2,
    )}\n`,
  );
});
