const fs = require("fs");
const path = require("path");

const htmlPath = path.join(__dirname, "..", "frontend", "index.html");
const jsPath = path.join(__dirname, "..", "frontend", "script.js");

const htmlContent = fs.readFileSync(htmlPath, "utf-8");
const jsContent = fs.readFileSync(jsPath, "utf-8");

console.log("=================================================");
console.log("FRONTEND NAVIGATION VERIFICATION SUITE");
console.log("=================================================");

// 1. Static HTML checks
const navItemRegex = /<a\s+class="nav-item[^"]*"\s+href="#([^"]+)"\s+data-tab="([^"]+)"\s+id="([^"]+)">/g;
const navItems = [];
let match;
while ((match = navItemRegex.exec(htmlContent)) !== null) {
    navItems.push({ hrefTab: match[1], dataTab: match[2], id: match[3] });
}

console.log(`[CHECK 1] Found ${navItems.length} sidebar navigation items in index.html:`);
navItems.forEach(item => {
    console.log(`  - Tab: "${item.dataTab}" (id: #${item.id}, href: #${item.hrefTab})`);
    if (item.hrefTab !== item.dataTab) {
        throw new Error(`Mismatch between href #${item.hrefTab} and data-tab "${item.dataTab}"!`);
    }
});

if (navItems.length !== 10) {
    throw new Error(`Expected 10 nav items, found ${navItems.length}`);
}
console.log("  => All nav item href and data-tab attributes match. [PASS]");

// Pane checks
const paneRegex = /<div\s+class="tab-pane\s+([^"]+)"\s+id="(pane-[^"]+)">/g;
const panes = [];
while ((match = paneRegex.exec(htmlContent)) !== null) {
    panes.push({ classes: match[1].split(/\s+/), id: match[2] });
}

console.log(`\n[CHECK 2] Found ${panes.length} tab panes in index.html:`);
panes.forEach(pane => {
    console.log(`  - Pane: #${pane.id} [classes: ${pane.classes.join(" ")}]`);
});

if (panes.length !== 10) {
    throw new Error(`Expected 10 tab panes, found ${panes.length}`);
}

navItems.forEach(item => {
    const expectedPaneId = `pane-${item.dataTab}`;
    const found = panes.find(p => p.id === expectedPaneId);
    if (!found) {
        throw new Error(`No matching pane #${expectedPaneId} found for nav item "${item.dataTab}"!`);
    }
});
console.log("  => Every nav item has a corresponding tab pane ID. [PASS]");

// Initial state check
const homePane = panes.find(p => p.id === "pane-home");
if (!homePane.classes.includes("active") || homePane.classes.includes("hidden")) {
    throw new Error("Initial state: #pane-home must be active and not hidden!");
}

const otherPanes = panes.filter(p => p.id !== "pane-home");
otherPanes.forEach(p => {
    if (!p.classes.includes("hidden") || p.classes.includes("active")) {
        throw new Error(`Initial state: #${p.id} must have class 'hidden' and not 'active'!`);
    }
});
console.log("  => Initial pane states correct (#pane-home is active; others are hidden). [PASS]");

// 2. Dynamic JS Navigation Logic Simulation
console.log("\n[CHECK 3] Simulating JavaScript Navigation Engine & Events...");

// DOM ClassList Mock
class ClassList {
    constructor(initial = []) {
        this.classes = new Set(initial);
    }
    add(cls) { this.classes.add(cls); }
    remove(cls) { this.classes.delete(cls); }
    contains(cls) { return this.classes.has(cls); }
    toggle(cls, force) {
        if (force !== undefined) {
            if (force) this.classes.add(cls);
            else this.classes.delete(cls);
            return force;
        }
        if (this.classes.has(cls)) {
            this.classes.delete(cls);
            return false;
        } else {
            this.classes.add(cls);
            return true;
        }
    }
    toString() { return Array.from(this.classes).join(" "); }
}

// DOM Element Mock
class MockElement {
    constructor(id, tag = "div", initialClasses = []) {
        this.id = id;
        this.tagName = tag.toUpperCase();
        this.classList = new ClassList(initialClasses);
        this.attributes = {};
        this.listeners = {};
        this.textContent = "";
        this.innerHTML = "";
        this.style = {};
        this.value = "";
    }
    setAttribute(k, v) { this.attributes[k] = v; }
    getAttribute(k) { return this.attributes[k]; }
    addEventListener(evt, fn) {
        if (!this.listeners[evt]) this.listeners[evt] = [];
        this.listeners[evt].push(fn);
    }
    click() {
        const event = {
            preventDefault: () => {},
            stopPropagation: () => {},
            target: this
        };
        (this.listeners["click"] || []).forEach(fn => fn(event));
    }
}

// Build mock document
const elementsById = {};
const mockPanes = panes.map(p => {
    const el = new MockElement(p.id, "div", p.classes);
    elementsById[p.id] = el;
    return el;
});

const mockNavItems = navItems.map(item => {
    const initialClasses = item.dataTab === "home" ? ["nav-item", "active"] : ["nav-item"];
    const el = new MockElement(item.id, "a", initialClasses);
    el.setAttribute("href", `#${item.hrefTab}`);
    el.setAttribute("data-tab", item.dataTab);
    elementsById[item.id] = el;
    return el;
});

const brandLink = new MockElement("brandLink", "a");
brandLink.setAttribute("href", "#home");
elementsById["brandLink"] = brandLink;

const breadcrumbSection = new MockElement("breadcrumbSection", "span");
breadcrumbSection.textContent = "Home";
elementsById["breadcrumbSection"] = breadcrumbSection;

const breadcrumbPage = new MockElement("breadcrumbPage", "span");
breadcrumbPage.textContent = "Check a Voice Recording";
elementsById["breadcrumbPage"] = breadcrumbPage;

const windowListeners = {};
const mockWindow = {
    location: { hash: "" },
    history: {
        pushState: (state, title, url) => {
            mockWindow.location.hash = url;
        }
    },
    addEventListener: (evt, fn) => {
        if (!windowListeners[evt]) windowListeners[evt] = [];
        windowListeners[evt].push(fn);
    },
    dispatchEvent: (evt) => {
        (windowListeners[evt.type] || []).forEach(fn => fn(evt));
    },
    scrollTo: () => {}
};

const mockDocument = {
    getElementById: (id) => elementsById[id] || null,
    querySelectorAll: (sel) => {
        if (sel === ".nav-item") return mockNavItems;
        if (sel === ".tab-pane") return mockPanes;
        return [];
    },
    body: { classList: new ClassList() }
};

// Extract navigation code block from frontend/script.js
const navCodeStart = jsContent.indexOf("// 1. TAB NAVIGATION");
const navCodeEnd = jsContent.indexOf("// -------------------------------------------------------------\n// 2. FILE HELPERS");
if (navCodeStart === -1 || navCodeEnd === -1) {
    throw new Error("Could not delineate TAB NAVIGATION section in script.js");
}
const navCode = jsContent.substring(navCodeStart, navCodeEnd);

// Wrap and evaluate in mock scope
const runNavCode = new Function(
    "document", "window", "_mockNavItems", "_mockTabPanes", "breadcrumbPage", "breadcrumbSection",
    "loadHistory", "populateCompareSelectors", "loadModelEval", "loadDatasetInfo", "loadSettings", "checkSystemHealth",
    navCode
);

const calls = {
    loadHistory: 0,
    populateCompareSelectors: 0,
    loadModelEval: 0,
    loadDatasetInfo: 0,
    loadSettings: 0,
    checkSystemHealth: 0
};

runNavCode(
    mockDocument,
    mockWindow,
    mockNavItems,
    mockPanes,
    breadcrumbPage,
    breadcrumbSection,
    () => { calls.loadHistory++; },
    () => { calls.populateCompareSelectors++; },
    () => { calls.loadModelEval++; },
    () => { calls.loadDatasetInfo++; },
    () => { calls.loadSettings++; },
    () => { calls.checkSystemHealth++; }
);

console.log("  => Tab navigation code loaded and attached event listeners successfully.");

// 3. Test clicking every single navigation item
console.log("\n[CHECK 4] Testing clicks on all 10 sidebar navigation items...");
navItems.forEach(item => {
    const navEl = elementsById[item.id];
    navEl.click();

    // Verify pane state
    const targetPaneId = `pane-${item.dataTab}`;
    const targetPane = elementsById[targetPaneId];

    if (targetPane.classList.contains("hidden")) {
        throw new Error(`Clicking ${item.dataTab}: Target pane ${targetPaneId} is still hidden!`);
    }
    if (!targetPane.classList.contains("active")) {
        throw new Error(`Clicking ${item.dataTab}: Target pane ${targetPaneId} is not active!`);
    }

    // Verify all other 9 panes are hidden
    mockPanes.forEach(pane => {
        if (pane.id !== targetPaneId) {
            if (!pane.classList.contains("hidden")) {
                throw new Error(`Clicking ${item.dataTab}: Other pane ${pane.id} is NOT hidden!`);
            }
            if (pane.classList.contains("active")) {
                throw new Error(`Clicking ${item.dataTab}: Other pane ${pane.id} is active!`);
            }
        }
    });

    // Verify nav item active class
    mockNavItems.forEach(n => {
        const isCurrent = n.getAttribute("data-tab") === item.dataTab;
        if (n.classList.contains("active") !== isCurrent) {
            throw new Error(`Clicking ${item.dataTab}: Nav item ${n.id} active state is wrong!`);
        }
    });

    // Verify window.location.hash
    if (mockWindow.location.hash !== `#${item.dataTab}`) {
        throw new Error(`Clicking ${item.dataTab}: window.location.hash is "${mockWindow.location.hash}", expected "#${item.dataTab}"`);
    }

    console.log(`  [PASS] Clicked "${item.dataTab}": #${targetPaneId} is visible, all others hidden, hash updated to #${item.dataTab}`);
});

// 4. Test Brand Link Click
console.log("\n[CHECK 5] Testing Brand Link Click (VoiceCheck Logo)...");
brandLink.click();
if (elementsById["pane-home"].classList.contains("hidden") || !elementsById["pane-home"].classList.contains("active")) {
    throw new Error("Brand link click did not activate #pane-home!");
}
console.log("  [PASS] Brand link click correctly activated #pane-home.");

// 5. Test Hash Changes
console.log("\n[CHECK 6] Testing Hash Navigation & Aliases...");
const testHashes = [
    { hash: "#history", expectedPane: "pane-history" },
    { hash: "#dashboard", expectedPane: "pane-home" },         // alias
    { hash: "#analysis", expectedPane: "pane-home" },          // alias
    { hash: "#reports", expectedPane: "pane-history" },        // alias
    { hash: "#evaluation", expectedPane: "pane-model-eval" },  // alias
    { hash: "#batch-analysis", expectedPane: "pane-batch" },   // alias
    { hash: "#unknown-random-hash", expectedPane: "pane-home" } // fallback
];

testHashes.forEach(t => {
    mockWindow.location.hash = t.hash;
    mockWindow.dispatchEvent({ type: "hashchange" });

    const activePane = mockPanes.find(p => !p.classList.contains("hidden"));
    if (!activePane || activePane.id !== t.expectedPane) {
        throw new Error(`Hash "${t.hash}": Expected #${t.expectedPane} to be visible, but active pane is #${activePane ? activePane.id : "none"}`);
    }
    console.log(`  [PASS] Hash "${t.hash}" correctly activated #${t.expectedPane}`);
});

// 6. Test Data-loader triggers
console.log("\n[CHECK 7] Verifying Tab-specific data loader invocations:");
console.log(`  - loadHistory calls: ${calls.loadHistory}`);
console.log(`  - populateCompareSelectors calls: ${calls.populateCompareSelectors}`);
console.log(`  - loadModelEval calls: ${calls.loadModelEval}`);
console.log(`  - loadDatasetInfo calls: ${calls.loadDatasetInfo}`);
console.log(`  - loadSettings calls: ${calls.loadSettings}`);
console.log(`  - checkSystemHealth calls: ${calls.checkSystemHealth}`);

if (calls.loadHistory === 0 || calls.loadModelEval === 0) {
    throw new Error("Tab data loaders were not triggered during navigation!");
}

console.log("\n=================================================");
console.log("ALL FRONTEND NAVIGATION TESTS PASSED SUCCESSFULLY");
console.log("=================================================");
