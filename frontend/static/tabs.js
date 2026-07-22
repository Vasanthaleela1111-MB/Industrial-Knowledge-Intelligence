function activateTopTab(key) {
    const buttons = document.querySelectorAll(".tab-btn");
    const panels = document.querySelectorAll(".tab-panel");

    buttons.forEach(btn => {
        btn.classList.toggle("active", btn.dataset.tab === key);
    });

    panels.forEach(panel => {
        panel.classList.toggle("active", panel.id === "tab-" + key);
    });
}

function parseGraphData(canvas) {
    try {
        return {
            nodes: JSON.parse(canvas.dataset.nodes || "[]"),
            edges: JSON.parse(canvas.dataset.edges || "[]"),
        };
    } catch (_error) {
        return { nodes: [], edges: [] };
    }
}

// function renderKnowledgeGraph() {
//     const canvas = document.getElementById("graph-canvas");
//     if (!canvas || canvas.dataset.rendered === "true") return;

//     const { nodes, edges } = parseGraphData(canvas);
//     canvas.dataset.rendered = "true";

//     if (!nodes.length) {
//         canvas.innerHTML = '<div class="alert alert-info">No graph nodes yet. Upload documents to build the knowledge graph.</div>';
//         return;
//     }

//     const width = Math.max(canvas.clientWidth || 900, 700);
//     const height = 600;
//     const cx = width / 2;
//     const cy = height / 2;
//     const radius = Math.min(width, height) * 0.35;

//     // Color coding for entity types
//     const typeColors = {
//         "Equipment": "#2563eb",      // Blue
//         "Component": "#7c3aed",     // Purple
//         "FailureMode": "#dc2626",   // Red
//         "MaintenanceActivity": "#ea580c", // Orange
//         "Inspection": "#0891b2",     // Cyan
//         "Standard": "#059669",       // Green
//         "Regulation": "#16a34a",     // Dark Green
//         "ProcessParameter": "#db2777", // Pink
//         "Hazard": "#f59e0b",         // Amber
//         "SafetyRequirement": "#ca8a04", // Yellow
//         "QualityRequirement": "#8b5cf6", // Violet
//         "default": "#64748b"         // Gray
//     };

//     const positions = new Map();
    
//     // Group nodes by type for better layout
//     const nodesByType = {};
//     nodes.forEach(node => {
//         const type = node.type || "default";
//         if (!nodesByType[type]) nodesByType[type] = [];
//         nodesByType[type].push(node);
//     });

//     const types = Object.keys(nodesByType);
//     let nodeIndex = 0;

//     types.forEach((type, typeIndex) => {
//         const typeNodes = nodesByType[type];
//         const typeAngle = (Math.PI * 2 * typeIndex) / types.length;
//         const typeRadius = radius * 0.7;
        
//         typeNodes.forEach((node, nodeIdx) => {
//             const angle = typeAngle + (nodeIdx / typeNodes.length) * (Math.PI * 2 / types.length);
//             const distance = typeRadius + (nodeIdx * 15);
            
//             positions.set(node.id, {
//                 x: cx + Math.cos(angle) * distance,
//                 y: cy + Math.sin(angle) * distance,
//             });
//             nodeIndex++;
//         });
//     });

//     const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
//     svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
//     svg.setAttribute("role", "img");
//     svg.setAttribute("aria-label", "Knowledge graph");

//     // Draw edges first (behind nodes)
//     edges.slice(0, 150).forEach(edge => {
//         const source = positions.get(edge.source);
//         const target = positions.get(edge.target);
//         if (!source || !target) return;
//         const line = document.createElementNS(svg.namespaceURI, "line");
//         line.setAttribute("x1", source.x);
//         line.setAttribute("y1", source.y);
//         line.setAttribute("x2", target.x);
//         line.setAttribute("y2", target.y);
//         line.setAttribute("stroke", "#94a3b8");
//         line.setAttribute("stroke-width", "1.5");
//         line.setAttribute("opacity", "0.6");
//         svg.appendChild(line);
//     });

//     // Draw nodes
//     nodes.slice(0, 100).forEach(node => {
//         const point = positions.get(node.id);
//         if (!point) return;
        
//         const type = node.type || "default";
//         const color = typeColors[type] || typeColors.default;
        
//         const group = document.createElementNS(svg.namespaceURI, "g");

//         // Node circle
//         const circle = document.createElementNS(svg.namespaceURI, "circle");
//         circle.setAttribute("cx", point.x);
//         circle.setAttribute("cy", point.y);
//         circle.setAttribute("r", "22");
//         circle.setAttribute("fill", color);
//         circle.setAttribute("stroke", "#ffffff");
//         circle.setAttribute("stroke-width", "2.5");
//         circle.setAttribute("cursor", "pointer");
//         group.appendChild(circle);

//         // Node label with better truncation
//         const label = document.createElementNS(svg.namespaceURI, "text");
//         label.setAttribute("x", point.x);
//         label.setAttribute("y", point.y + 38);
//         label.setAttribute("text-anchor", "middle");
//         label.setAttribute("font-size", "11");
//         label.setAttribute("font-weight", "500");
//         label.setAttribute("fill", "#334155");
        
//         const labelText = String(node.label || node.id);
//         // Smart truncation: show first 15 chars, then ... if longer
//         const displayLabel = labelText.length > 18 ? labelText.slice(0, 15) + "..." : labelText;
//         label.textContent = displayLabel;
//         group.appendChild(label);

//         // Tooltip with full information
//         const title = document.createElementNS(svg.namespaceURI, "title");
//         title.textContent = `${node.label || node.id}\nType: ${node.type || "Entity"}`;
//         group.appendChild(title);
        
//         svg.appendChild(group);
//     });

//     canvas.innerHTML = "";
//     canvas.appendChild(svg);
// }

function renderKnowledgeGraph() {

    const canvas = document.getElementById("graph-canvas");

    if (!canvas || canvas.dataset.rendered === "true")
        return;

    canvas.dataset.rendered = "true";

    const graph = parseGraphData(canvas);

    if (!graph.nodes.length) {

        canvas.innerHTML =
            "<div class='alert alert-info'>No graph available.</div>";

        return;
    }
    console.log(graph.nodes);

    const nodes = new vis.DataSet(

        graph.nodes.map(node => ({

            id: node.id,

            label: node.label || node.id,

            group: node.type || "Entity",

            title:
                "<b>" + (node.label || node.id) +
                "</b><br>" +
                (node.type || "Entity"),

            shape:
                node.type === "Document"
                    ? "box"
                    : "dot",

            size:
                node.type === "Document"
                    ? 35
                    : 20

        }))

    );

    const edges = new vis.DataSet(

        graph.edges.map(edge => ({

            from: edge.source,

            to: edge.target,

            label: edge.relation || "",

            arrows: "to",

            font: {

                size: 12,

                align: "middle"

            },

            color: {

                color: "#94a3b8"

            },

            smooth: true

        }))

    );

    const options = {

        physics: {

            enabled: true,

            stabilization: true,

            barnesHut: {

                gravitationalConstant: -9000,

                springLength: 180,

                springConstant: 0.02

            }

        },

        interaction: {

            hover: true,

            navigationButtons: true,

            zoomView: true,

            dragView: true

        },

        nodes: {

            font: {

                size: 16,

                face: "Arial"

            },

            borderWidth: 2

        },

        groups: {

            Document: {

                color: {

                    background: "#2563eb"

                },

                shape: "box"

            },

            Equipment: {

                color: {

                    background: "#16a34a"

                }

            },

            Component: {

                color: {

                    background: "#10b981"

                }

            },

            Standard: {

                color: {

                    background: "#7c3aed"

                }

            },

            FailureMode: {

                color: {

                    background: "#dc2626"

                }

            },

            RootCause: {

                color: {

                    background: "#ea580c"

                }

            },

            Entity: {

                color: {

                    background: "#64748b"

                }

            }

        }

    };

    canvas.innerHTML = "";

    new vis.Network(
        canvas,
        {
            nodes,
            edges
        },
        options
    );

}


document.addEventListener("DOMContentLoaded", () => {
    const buttons = document.querySelectorAll(".tab-btn");

    buttons.forEach(button => {
        button.addEventListener("click", function () {
            activateTopTab(this.dataset.tab);
            if (this.dataset.tab === "assistant") renderKnowledgeGraph();
        });
    });

    const initialTab = window.__activeTab || "workspace";
    activateTopTab(initialTab);
    if (initialTab === "assistant") renderKnowledgeGraph();
});