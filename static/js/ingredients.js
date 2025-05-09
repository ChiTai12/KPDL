/* @ts-nocheck */
// Dữ liệu cho sơ đồ mạng
const data = {
  nodes: [],
  links: [],
};

// Lấy dữ liệu từ biến toàn cục
const frequentItemsets = window.ingredientData || [];
console.log("Dữ liệu nhận được:", window.ingredientData);
console.log("Frequent itemsets:", frequentItemsets);

// Tạo nodes từ các tập nguyên liệu
const ingredientSet = new Set();

// Thêm tất cả các nguyên liệu vào set
frequentItemsets.forEach((item) => {
  item.itemsets.forEach((ingredient) => {
    ingredientSet.add(ingredient);
  });
});

// Thêm nodes
ingredientSet.forEach((ingredient) => {
  data.nodes.push({
    id: ingredient,
    group: 1,
    size: 1,
  });
});

// Thêm links từ các tập phổ biến
frequentItemsets.forEach((item) => {
  if (item.itemsets.length > 1) {
    for (let i = 0; i < item.itemsets.length; i++) {
      for (let j = i + 1; j < item.itemsets.length; j++) {
        data.links.push({
          source: item.itemsets[i],
          target: item.itemsets[j],
          value: item.support,
          strength: item.support,
        });
      }
    }
  }
});

// Lấy SVG container có sẵn
const svg = d3.select("#network-graph svg");
const width = 800;
const height = 600;

// Tạo container cho graph
const g = svg.append("g");

// Tạo simulation
const simulation = d3
  .forceSimulation(data.nodes)
  .force(
    "link",
    d3
      .forceLink(data.links)
      .id((d) => d.id)
      .distance(100)
      .strength((d) => d.strength)
  )
  .force("charge", d3.forceManyBody().strength(-300))
  .force("center", d3.forceCenter(width / 2, height / 2))
  .force("collision", d3.forceCollide().radius(30));

// Vẽ links
const link = g
  .append("g")
  .attr("class", "links")
  .selectAll("line")
  .data(data.links)
  .join("line")
  .attr("stroke", "#999")
  .attr("stroke-opacity", 0.6)
  .attr("stroke-width", (d) => Math.sqrt(d.value) * 2);

// Vẽ nodes
const node = g
  .append("g")
  .attr("class", "nodes")
  .selectAll("circle")
  .data(data.nodes)
  .join("circle")
  .attr("r", 8)
  .attr("fill", "#69b3a2")
  .call(drag(simulation));

// Thêm labels
const labels = g
  .append("g")
  .attr("class", "labels")
  .selectAll("text")
  .data(data.nodes)
  .join("text")
  .text((d) => d.id)
  .attr("font-size", "10px")
  .attr("dx", 12)
  .attr("dy", 4);

// Cập nhật vị trí
simulation.on("tick", () => {
  link
    .attr("x1", (d) => d.source.x)
    .attr("y1", (d) => d.source.y)
    .attr("x2", (d) => d.target.x)
    .attr("y2", (d) => d.target.y);

  node.attr("cx", (d) => d.x).attr("cy", (d) => d.y);

  labels.attr("x", (d) => d.x).attr("y", (d) => d.y);
});

// Thêm tính năng kéo thả
function drag(simulation) {
  function dragstarted(event) {
    if (!event.active) simulation.alphaTarget(0.3).restart();
    event.subject.fx = event.subject.x;
    event.subject.fy = event.subject.y;
  }

  function dragged(event) {
    event.subject.fx = event.x;
    event.subject.fy = event.y;
  }

  function dragended(event) {
    if (!event.active) simulation.alphaTarget(0);
    event.subject.fx = null;
    event.subject.fy = null;
  }

  return d3
    .drag()
    .on("start", dragstarted)
    .on("drag", dragged)
    .on("end", dragended);
}

// Thêm tính năng zoom
const zoom = d3
  .zoom()
  .scaleExtent([0.1, 4])
  .on("zoom", (event) => {
    g.attr("transform", event.transform);
  });

svg.call(zoom);
