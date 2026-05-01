/* global mermaid */

(() => {
  const API_JSON_PATH = "api-reference.json";

  const copyButtons = document.querySelectorAll(".copy-btn");
  copyButtons.forEach((button) => {
    button.addEventListener("click", async () => {
      const target = document.querySelector(button.getAttribute("data-copy"));
      if (!target) {
        return;
      }
      const text = target.textContent || "";
      try {
        await navigator.clipboard.writeText(text);
        const oldText = button.textContent;
        button.textContent = "Copied";
        setTimeout(() => {
          button.textContent = oldText;
        }, 1200);
      } catch (_error) {
        button.textContent = "Copy failed";
        setTimeout(() => {
          button.textContent = "Copy";
        }, 1200);
      }
    });
  });

  const apiContent = document.getElementById("api-content");

  const renderList = (items = []) => {
    if (!items.length) {
      return "<p class=\"empty\">None</p>";
    }
    return `<ul>${items.map((item) => `<li>${item}</li>`).join("")}</ul>`;
  };

  const renderParams = (params = []) => {
    if (!params.length) {
      return "<p class=\"empty\">No parameters</p>";
    }
    return `<ul>${params
      .map(
        (param) =>
          `<li><code>${param.name}</code> <span class=\"param-type\">${param.type}</span>: ${param.description}</li>`
      )
      .join("")}</ul>`;
  };

  const buildFunctionCard = (func) => {
    const paramsHtml = renderParams(func.params);
    const sideEffectsHtml = renderList(func.sideEffects);
    const notesHtml = renderList(func.implementationNotes);

    return `
      <details class="api-item">
        <summary>
          <span class="api-signature">${func.signature}</span>
          <span class="pill">${func.name}</span>
        </summary>
        <p class="api-summary">${func.summary}</p>
        <div class="api-grid">
          <div>
            <h4>Parameters</h4>
            ${paramsHtml}
          </div>
          <div>
            <h4>Returns</h4>
            <p>${func.returns || "None"}</p>
            <h4>Side Effects</h4>
            ${sideEffectsHtml}
          </div>
        </div>
        <h4>Implementation Notes</h4>
        ${notesHtml}
      </details>
    `;
  };

  const renderApi = (data) => {
    if (!apiContent) {
      return;
    }

    const modules = Array.isArray(data.modules) ? data.modules : [];
    if (!modules.length) {
      apiContent.innerHTML = '<p class="empty">No API modules found in JSON.</p>';
      return;
    }

    apiContent.innerHTML = modules
      .map((module) => {
        const patternPills = (module.patterns || [])
          .map((pattern) => `<span class="pill pill-soft">${pattern}</span>`)
          .join("");

        const functions = (module.functions || []).map((func) => buildFunctionCard(func)).join("");

        return `
          <section class="api-group" data-group="${module.module}">
            <div class="api-module-head">
              <h3>${module.module}</h3>
              <p>${module.role || ""}</p>
              <div class="pill-row">${patternPills}</div>
            </div>
            ${functions}
          </section>
        `;
      })
      .join("");

    bindSearchLogic("api-search", ".api-group", ".api-item");
  };

  const renderGraphs = (data) => {
    const graphContent = document.getElementById("graph-content");
    if (!graphContent || !Array.isArray(data)) return;

    graphContent.innerHTML = data
      .map((tab) => {
        const graphsHtml = (tab.graphs || []).map((graph) => `
          <details class="api-item graph-item">
            <summary>
              <span class="api-signature">${graph.title}</span>
              <span class="pill pill-soft">${graph.metric}</span>
            </summary>
            <p class="api-summary">${graph.description}</p>
            <div class="api-grid" style="margin-top: 1rem;">
              <div>
                <strong style="font-size: 0.875rem; text-transform: uppercase; color: var(--text-muted); display: block; margin-bottom: 0.5rem">Primary Data Source</strong>
                <p style="background: var(--surface-hover); padding: 0.5rem; border-radius: var(--radius-sm); border: 1px solid var(--border-color);">${graph.source}</p>
              </div>
            </div>
          </details>
        `).join("");

        return `
          <section class="api-group graph-group">
            <div class="api-module-head">
              <h3>${tab.tab}</h3>
              <p>${tab.description || ""}</p>
            </div>
            ${graphsHtml || '<p class="empty" style="margin-left: 1.5rem">No top-level graphs in this tab.</p>'}
          </section>
        `;
      })
      .join("");

    bindSearchLogic("graph-search", ".graph-group", ".graph-item");
  };

  const bindSearchLogic = (inputId, groupSelector, itemSelector) => {
    const searchInput = document.getElementById(inputId);
    if (!searchInput) return;

    const groups = Array.from(document.querySelectorAll(groupSelector));
    const allItems = Array.from(document.querySelectorAll(itemSelector));

    searchInput.addEventListener("input", () => {
      const query = searchInput.value.trim().toLowerCase();

      allItems.forEach((item) => {
        const visible = item.textContent.toLowerCase().includes(query);
        item.style.display = visible ? "" : "none";
      });

      groups.forEach((group) => {
        const hasVisibleItems = Array.from(group.querySelectorAll(itemSelector)).some(
          (item) => item.style.display !== "none"
        );
        group.style.display = hasVisibleItems ? "" : "none";
      });
    });
  };

  const loadApiReference = async () => {
    if (apiContent) {
      const fallbackData = typeof window !== "undefined" && window.API_REFERENCE_DATA ? window.API_REFERENCE_DATA : null;

      try {
        const response = await fetch(API_JSON_PATH);
        if (!response.ok) {
          throw new Error(`HTTP ${response.status}`);
        }
        const data = await response.json();
        renderApi(data);
      } catch (error) {
        if (fallbackData && Array.isArray(fallbackData.modules)) {
          renderApi(fallbackData);
        } else {
          apiContent.innerHTML = `
            <p class="empty">Could not load API JSON (${String(error)}).</p>
            <p class="small-note">Run a local static server in this folder (for example: <code>python -m http.server 8080</code>) and open the site via http://localhost.</p>
          `;
        }
      }
    }

    const graphData = typeof window !== "undefined" && window.GRAPH_REFERENCE_DATA ? window.GRAPH_REFERENCE_DATA : [];
    if (graphData.length > 0) {
      renderGraphs(graphData);
    }
  };

  loadApiReference();

  if (typeof mermaid !== "undefined") {
    mermaid.initialize({
      startOnLoad: true,
      theme: "base",
      themeVariables: {
        primaryColor: "#f8e9da",
        primaryTextColor: "#1f1b16",
        primaryBorderColor: "#c44d24",
        lineColor: "#6a6256",
        fontFamily: "Space Grotesk",
        fontSize: "14px"
      }
    });
  }
})();
