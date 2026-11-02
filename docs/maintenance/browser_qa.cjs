"use strict";
// Optional QA helper using a host-provided browser/runtime; no website npm setup.
const fs = require("node:fs");
const path = require("node:path");
const http = require("node:http");
const { spawn } = require("node:child_process");
const assert = require("node:assert/strict");
const root = path.resolve(__dirname, "..");
const bundled = path.resolve(path.dirname(process.execPath), "../node_modules/playwright");
const { chromium } = require(process.env.MEDSPAR_PLAYWRIGHT_PATH || bundled);
const executablePath = process.env.MEDSPAR_BROWSER_PATH || "C:/Program Files/Google/Chrome/Application/chrome.exe";
let base = process.env.MEDSPAR_PREVIEW_URL;
const report = { date: "2026-11-02", viewports: [], console_errors: [], failed_requests: [], assertions: [], svg_text_overflow: [] };
const check = (condition, label) => { assert(condition, label); report.assertions.push(label); };

async function main() {
  // Independently emulate GitHub Pages project-subpath hosting, without copying files.
  const subpath = http.createServer((req, res) => {
    const urlPath = decodeURIComponent(new URL(req.url, "http://localhost").pathname);
    if (!urlPath.startsWith("/MedSPAR/")) { res.writeHead(404).end(); return; }
    const file = path.resolve(root, urlPath.slice("/MedSPAR/".length) || "index.html");
    if (!file.startsWith(root + path.sep) || !fs.existsSync(file) || !fs.statSync(file).isFile()) { res.writeHead(404).end(); return; }
    const mime = { ".html": "text/html", ".css": "text/css", ".js": "text/javascript", ".svg": "image/svg+xml", ".png": "image/png", ".json": "application/json" };
    res.writeHead(200, { "Content-Type": mime[path.extname(file)] || "text/plain" });
    fs.createReadStream(file).pipe(res);
  });
  await new Promise(resolve => subpath.listen(0, "127.0.0.1", resolve));
  let browser;
  let preview;
  try {
    if (!base) {
      preview = spawn("python", ["-u", "-m", "http.server", "0", "--directory", root, "--bind", "127.0.0.1"], { windowsHide: true });
      base = await new Promise((resolve, reject) => {
        const timer = setTimeout(() => reject(new Error("Python preview startup timed out")), 10000);
        preview.on("error", error => { clearTimeout(timer); reject(error); });
        preview.on("exit", code => { clearTimeout(timer); reject(new Error(`Python preview exited: ${code}`)); });
        preview.stdout.on("data", chunk => {
          const match = String(chunk).match(/port (\d+)/);
          if (match) { clearTimeout(timer); resolve(`http://127.0.0.1:${match[1]}`); }
        });
      });
      report.preview = "Temporary python -m http.server on an automatically selected loopback port";
    }
    browser = await chromium.launch({ executablePath, headless: true, args: ["--disable-gpu", "--no-first-run"] });
    const context = await browser.newContext();
    const page = await context.newPage();
    page.on("pageerror", e => report.console_errors.push(e.message));
    page.on("console", e => { if (e.type() === "error") report.console_errors.push(e.text()); });
    page.on("requestfailed", req => report.failed_requests.push({ url: req.url(), failure: req.failure() }));
    for (const [name, width, height] of [["desktop", 1440, 1000], ["tablet", 768, 1024], ["mobile", 390, 844], ["small-mobile", 320, 740]]) {
      await page.setViewportSize({ width, height });
      const response = await page.goto(base, { waitUntil: "networkidle" });
      check(response.status() === 200, `${name}: HTTP 200`);
      for (const img of await page.locator("main img").all()) { await img.scrollIntoViewIfNeeded(); await img.evaluate(el => el.decode()); }
      const dimensions = await page.evaluate(() => ({ viewport: innerWidth, document: document.documentElement.scrollWidth, images: [...document.querySelectorAll("main img")].map(x => ({ src: x.getAttribute("src"), loaded: x.complete && x.naturalWidth > 0 })) }));
      check(dimensions.document <= width, `${name}: no horizontal document overflow`);
      check(dimensions.images.every(x => x.loaded), `${name}: all main images decoded`);
      await page.evaluate(() => window.scrollTo(0, 0));
      await page.screenshot({ path: path.join(root, `qa/${name}.png`), fullPage: true });
      report.viewports.push({ name, width, height, ...dimensions });
    }
    await page.locator('nav a[href="#method"]').click();
    check(new URL(page.url()).hash === "#method", "navigation reaches method hash");
    const trigger = page.getByRole("button", { name: "Enlarge figure" });
    await trigger.click();
    check(await page.locator("dialog").evaluate(el => el.open), "figure dialog opens");
    await page.keyboard.press("Escape");
    check(!(await page.locator("dialog").evaluate(el => el.open)), "Escape closes dialog");
    check(await trigger.evaluate(el => document.activeElement === el), "focus returns to dialog trigger");
    await trigger.click();
    await page.getByRole("button", { name: "Close enlarged figure" }).click();
    check(!(await page.locator("dialog").evaluate(el => el.open)), "Close button closes dialog");
    await page.locator("#resources details summary").click();
    await context.grantPermissions(["clipboard-read", "clipboard-write"], { origin: base });
    await page.getByRole("button", { name: "Copy command" }).click();
    await page.waitForFunction(() => document.querySelector("[data-copy]").textContent === "Copied");
    check(await page.evaluate(async () => (await navigator.clipboard.readText()).includes("python -m http.server 8000 --directory docs")), "clipboard copies the preview command");
    await page.emulateMedia({ reducedMotion: "reduce" });
    check(await page.evaluate(() => getComputedStyle(document.documentElement).scrollBehavior === "auto"), "reduced-motion disables smooth scrolling");
    const offline = await browser.newContext({ javaScriptEnabled: false });
    const nojs = await offline.newPage();
    await nojs.goto(base);
    check(await nojs.locator("#dataset table tbody tr").count() === 10, "no-JS: ten dataset rows remain");
    check(await nojs.locator("#method h2").isVisible(), "no-JS: method text visible");
    await offline.close();
    const projectURL = `http://127.0.0.1:${subpath.address().port}/MedSPAR/`;
    await page.goto(projectURL, { waitUntil: "networkidle" });
    for (const img of await page.locator("main img").all()) { await img.scrollIntoViewIfNeeded(); await img.evaluate(el => el.decode()); }
    check(await page.locator("main img").evaluateAll(imgs => imgs.every(x => x.complete && x.naturalWidth > 0 && x.src.includes("/MedSPAR/assets/"))), "project subpath: all images load under /MedSPAR/assets/");
    check(await page.locator("h1").evaluate(el => getComputedStyle(el).fontSize !== "32px"), "project subpath: CSS loads");
    check(await page.locator("[data-zoom]").isVisible(), "project subpath: JavaScript enhancement loads");
    for (const file of ["teaser/task-overview.svg", "teaser/task-overview-mobile.svg", "framework/method-overview.svg", "dataset/construction.svg", "dataset/construction-mobile.svg"]) {
      // Inspect inline SVG in a fresh page; raw SVG navigation causes an unrelated
      // browser-generated /favicon.ico request on Python's directory server.
      const svgPage = await context.newPage();
      const svgResponse = await context.request.get(`${base}/assets/images/${file}`);
      check(svgResponse.status() === 200, `${file}: SVG HTTP 200`);
      await svgPage.setContent(await svgResponse.text());
      const overflows = await svgPage.evaluate(() => {
        const svg = document.querySelector("svg"), v = svg.viewBox.baseVal;
        return [...svg.querySelectorAll("text")].filter(el => { const b = el.getBBox(); return b.x < 0 || b.y < 0 || b.x + b.width > v.width || b.y + b.height > v.height; }).map(el => el.textContent);
      });
      report.svg_text_overflow.push({ file, overflows });
      check(overflows.length === 0, `${file}: SVG text within canvas`);
      await svgPage.close();
    }
    check(report.console_errors.length === 0, "no browser console/page errors");
    check(report.failed_requests.length === 0, "no failed browser resource requests");
    report.passed = true;
  } catch (error) { report.passed = false; report.error = error.stack; throw error; }
  finally {
    if (browser) await browser.close();
    if (preview) preview.kill();
    await new Promise(resolve => subpath.close(resolve));
    fs.writeFileSync(path.join(root, "qa/browser_checks.json"), JSON.stringify(report, null, 2) + "\n");
  }
  console.log(JSON.stringify({ passed: report.passed, viewports: report.viewports.map(x => x.name), assertions: report.assertions.length }));
}
main().catch(error => { console.error(error.message); process.exitCode = 1; });
