// To modify this just run `typst watch camera_models.typ` and watch it update live
#import "@preview/fletcher:0.5.8" as fletcher: diagram, node, edge
#set page(width: auto, height: auto, margin: 1em, fill: white.transparentize(0%))
#let d1 = diagram(
    edge-stroke: 3pt,
    label-size: 3em,
    node((0,0), name: <ul>),
    node((10,0), name: <ur>),
    node((0,6), name: <ll>),
    node((10,6), name: <lr>),
    node((5,3), name: <center>),
    edge(<center>, (rel: (-2, 0)), "-}>", `u`, label-pos: 1, label-side: left),
    edge(<center>, (rel: (0, -2)), "-}>", `v`, label-pos: 1),
    node(enclose: ((0,0), (10, 0), (10, 6)), stroke: 1pt, inset: 0em, fill: teal.lighten(90%)),
    edge(<ll>, <lr>, `W`, stroke: 0pt, label-side: right),
    edge(<lr>, <ur>, `H`, stroke: 0pt, label-side: right),
    edge(<ul>, <ur>, `:center`, stroke: 0pt, label-side: left, label-wrapper: e=>box(e.label, fill: gray.lighten(50%), inset: 0.15em, radius: 0.1em, stroke: 1pt+black)),
)
#let d2 = diagram(
    edge-stroke: 3pt,
    label-size: 3em,
    node((0,0), name: <ul>),
    node((10,0), name: <ur>),
    node((0,6), name: <ll>),
    node((10,6), name: <lr>),
    node((0,0), name: <center>),
    node(enclose: ((0,0), (10, 0), (10, 6)), stroke: 1pt, inset: 0.001em, fill: teal.lighten(90%)),
    edge(<center>, (rel: (2, 0)), "-}>", `u`, label-pos: 1, label-side: right),
    edge(<center>, (rel: (0, 2)), "-}>", `v`, label-pos: 1, label-side: left),
    edge(<ll>, <lr>, `W`, stroke: 0pt, label-side: right),
    edge(<lr>, <ur>, `H`, stroke: 0pt, label-side: right),
    edge(<ul>, <ur>, `:offset`, stroke: 0pt, label-side: left, label-wrapper: e=>box(e.label, fill: gray.lighten(50%), inset: 0.15em, radius: 0.1em, stroke: 1pt+black)),
)
#grid(columns: (auto, auto),
    column-gutter: 10em,
    d1, d2
)
