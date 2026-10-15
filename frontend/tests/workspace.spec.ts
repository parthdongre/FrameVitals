import { test, expect } from "@playwright/test";
import AxeBuilder from "@axe-core/playwright";

test("CSV source → inspect → configure → real report → export", async ({
  page,
}) => {
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await page.goto("/#/analyze");
  await expect(
    page.getByRole("heading", { name: "One source. A clearer picture." }),
  ).toBeVisible();
  await page
    .getByLabel("Source file", { exact: true })
    .setInputFiles({
      name: "measurements.csv",
      mimeType: "text/csv",
      buffer: Buffer.from("value,group\n1,a\n2,b\n,a\n4,b\n"),
    });
  await expect(page.getByText("tabular detected")).toBeVisible();
  await expect(
    page.getByRole("columnheader", { name: "value", exact: true }),
  ).toBeVisible();
  await page.getByRole("radio", { name: "Quick" }).check();
  await page.screenshot({
    path: "test-results/workspace-desktop.png",
    fullPage: true,
  });
  let accessibility = await new AxeBuilder({ page }).include("main").analyze();
  expect(accessibility.violations).toEqual([]);
  await page.getByRole("button", { name: "Run analysis" }).click();
  await expect(
    page.getByRole("heading", { name: "Your source, understood." }),
  ).toBeVisible();
  await expect(
    page.getByText("measurements.csv", { exact: true }),
  ).toBeVisible();
  await expect(
    page.getByText("Missing values by column", { exact: true }),
  ).toBeVisible();
  await expect(page.getByText("25%", { exact: true })).toBeVisible();
  const download = page.waitForEvent("download");
  await page.getByRole("button", { name: "Export JSON" }).click();
  expect((await download).suggestedFilename()).toBe("framevitals-result.json");
  await page.screenshot({
    path: "test-results/report-desktop.png",
    fullPage: true,
  });
  accessibility = await new AxeBuilder({ page }).include("main").analyze();
  expect(accessibility.violations).toEqual([]);
  expect(errors).toEqual([]);
});

test("document and reference workflow on a narrow screen", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/#/analyze");
  await page
    .getByLabel("Source file", { exact: true })
    .setInputFiles({
      name: "notes.md",
      mimeType: "text/markdown",
      buffer: Buffer.from("# Analysis\nA useful source with text."),
    });
  await expect(page.getByText("document detected")).toBeVisible();
  await expect(page.getByLabel("Focus column")).toHaveCount(0);
  await page
    .getByText("Compare with a reference (optional)", { exact: true })
    .click();
  await page
    .getByLabel("Reference file")
    .setInputFiles({
      name: "before.md",
      mimeType: "text/markdown",
      buffer: Buffer.from("# Before\nEarlier version."),
    });
  await page.getByRole("radio", { name: "Quick" }).check();
  await page.getByRole("button", { name: "Run analysis" }).click();
  await expect(
    page.getByRole("heading", { name: "Your source, understood." }),
  ).toBeVisible();
  await expect(
    page.getByRole("heading", { name: "Reference comparison" }),
  ).toBeVisible();
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= window.innerWidth,
    ),
  ).toBe(true);
  await page.screenshot({
    path: "test-results/report-mobile.png",
    fullPage: true,
  });
  const accessibility = await new AxeBuilder({ page })
    .include("main")
    .analyze();
  expect(accessibility.violations).toEqual([]);
});

test("unsafe input produces an actionable error without navigating", async ({
  page,
}) => {
  await page.goto("/#/analyze");
  await page
    .getByLabel("Source file", { exact: true })
    .setInputFiles({
      name: "unsafe.pkl",
      mimeType: "application/octet-stream",
      buffer: Buffer.from("no"),
    });
  await expect(page.getByRole("alert")).toContainText("Unsupported file");
  await expect(
    page.getByRole("button", { name: "Run analysis" }),
  ).toBeDisabled();
});

test('graph and tensor results expose source-specific measured views', async ({ page }) => {
  await page.goto('/#/analyze');
  const graph = '<?xml version="1.0"?><graphml xmlns="http://graphml.graphdrawing.org/xmlns"><graph edgedefault="undirected"><node id="alpha"/><node id="beta"/><edge source="alpha" target="beta"/></graph></graphml>';
  await page.getByLabel('Source file', { exact: true }).setInputFiles({ name: 'network.graphml', mimeType: 'application/xml', buffer: Buffer.from(graph) });
  await expect(page.getByText('graph detected')).toBeVisible();
  await page.getByRole('radio', { name: 'Quick' }).check();
  await page.getByRole('button', { name: 'Run analysis' }).click();
  await expect(page.getByRole('heading', { name: 'Network excerpt' })).toBeVisible();
  await expect(page.getByRole('img', { name: /Graph excerpt with 2 nodes and 1 edges/ })).toBeVisible();
  expect((await new AxeBuilder({ page }).include('main').analyze()).violations).toEqual([]);
  await page.getByRole('button', { name: 'New analysis' }).click();
  // Small v1 NPY numeric fixture; never contains Python objects.
  const rawHeader = "{'descr': '<f8', 'fortran_order': False, 'shape': (2, 2), }";
  const header = rawHeader + ' '.repeat((64 - (10 + rawHeader.length + 1) % 64) % 64) + '\n';
  const prefix = Buffer.alloc(10); prefix.set([0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59, 1, 0]); prefix.writeUInt16LE(header.length, 8);
  const values = Buffer.alloc(32); [1, 0, 0, 1].forEach((v, i) => values.writeDoubleLE(v, i * 8));
  await page.getByLabel('Source file', { exact: true }).setInputFiles({ name: 'matrix.npy', mimeType: 'application/octet-stream', buffer: Buffer.concat([prefix, Buffer.from(header), values]) });
  await expect(page.getByText('tensor detected')).toBeVisible();
  await page.getByRole('radio', { name: 'Quick' }).check();
  await page.getByRole('button', { name: 'Run analysis' }).click();
  await expect(page.getByRole('heading', { name: 'Singular values (up to 24)' })).toBeVisible();
  await expect(page.getByRole('heading', { name: 'Value diagnostics' })).toBeVisible();
  expect((await new AxeBuilder({ page }).include('main').analyze()).violations).toEqual([]);
});
