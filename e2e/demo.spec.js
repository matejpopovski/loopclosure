const { test, expect } = require("@playwright/test");

const finished = () => window.loopclosure.finished === true;

function trackProblems(page) {
  const problems = [];
  page.on("pageerror", (error) => problems.push(error.message));
  page.on("console", (message) => {
    if (message.type() === "error") problems.push(message.text());
  });
  return problems;
}

async function paintedShare(page, id) {
  return page.evaluate((canvasId) => {
    const canvas = document.getElementById(canvasId);
    const { width, height } = canvas;
    const pixels = canvas.getContext("2d").getImageData(0, 0, width, height).data;
    let bright = 0;
    for (let i = 0; i < pixels.length; i += 4) {
      if (pixels[i] + pixels[i + 1] + pixels[i + 2] > 300) bright += 1;
    }
    return bright / (width * height);
  }, id);
}

test("a simulated run builds both maps and closure beats drift", async ({ page }) => {
  const problems = trackProblems(page);
  await page.goto("index.html?input=atrium&seed=3&speed=1000");
  await page.waitForFunction(finished);

  const result = await page.evaluate(() => window.loopclosure);
  expect(result.error).toBeNull();
  expect(result.keyframes).toBeGreaterThan(400);
  expect(result.closures).toBeGreaterThan(10);
  expect(await paintedShare(page, "open")).toBeGreaterThan(0.002);
  expect(await paintedShare(page, "closed")).toBeGreaterThan(0.002);
  expect(result.errorClosed).toBeLessThan(0.5);
  expect(problems).toEqual([]);
});

test("the real log replays from its compressed file", async ({ page }) => {
  const problems = trackProblems(page);
  await page.goto("index.html?input=intel&speed=1000");
  await page.waitForFunction(finished);

  const result = await page.evaluate(() => window.loopclosure);
  expect(result.keyframes).toBeGreaterThan(2000);
  expect(result.closures).toBeGreaterThan(100);
  await expect(page.locator("#label-path")).toHaveText("Odometric path");
  await expect(page.locator("#stat-open")).toBeHidden();
  await expect(page.locator("#chart")).toBeHidden();
  await expect(page.locator("#seed")).toBeHidden();
  expect(problems).toEqual([]);
});

test("pause stops the run and restart begins it again", async ({ page }) => {
  await page.goto("index.html?input=atrium&speed=4");
  await page.waitForFunction(() => window.loopclosure.keyframes > 20);
  await page.getByRole("button", { name: "Pause" }).click();
  await page.waitForTimeout(100);
  const paused = await page.evaluate(() => window.loopclosure.keyframes);
  await page.waitForTimeout(400);
  expect(await page.evaluate(() => window.loopclosure.keyframes)).toBe(paused);

  await page.getByRole("button", { name: "Play" }).click();
  await page.waitForFunction((n) => window.loopclosure.keyframes > n, paused);
  await page.getByRole("button", { name: "Restart" }).click();
  await page.waitForFunction((n) => window.loopclosure.keyframes < n, paused);
});

test("the chart answers hover with values at that distance", async ({ page }) => {
  await page.goto("index.html?input=atrium&speed=1000");
  await page.waitForFunction(finished);
  await page.locator("#chart").hover({ position: { x: 400, y: 60 } });
  await expect(page.locator("#tip")).toContainText(/m driven: .* without, .* with/);
});

test("a phone-width page does not scroll sideways", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("index.html?input=atrium&speed=4");
  await page.waitForFunction(() => window.loopclosure.keyframes > 5);
  const overflow = await page.evaluate(
    () => document.documentElement.scrollWidth - document.documentElement.clientWidth,
  );
  expect(overflow).toBeLessThanOrEqual(0);
});
