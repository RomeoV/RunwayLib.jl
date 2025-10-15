#import "@preview/cetz:0.4.2"
#set page(width: auto, height: auto, margin: 0.5em)
#cetz.canvas({
    let x1 = 2.4
    let x2 = 3.8
    let y1 = 2
    let y2 = 3.5
    let x3 = 4.5
    let x4 = 3.3
    let W = 8
    let H = 6
  import cetz.draw: *
    rect((0,0), (W, H), fill: teal.lighten(90%))
    line((0,0), (W, 0), (W, H), (0, H), close: true, stroke: 0.5pt)
    line((x1, 0), (x2, H), name: "lleft", stroke: 0.5pt)
    line((x3, 0), (x4, H), name: "lright", stroke: 0.5pt)
    line((0, y1), (W, y1), stroke: (thickness: 0.5pt, dash: "dashed"), name: "lhbot")
    line((0, y2), (W, y2), stroke: (thickness: 0.5pt, dash: "dashed"), name: "lhtop")

    intersections("i", "lleft", "lhtop")
    circle("i.0", radius: 3pt, fill: red)
    intersections("j", "lright", "lhtop")
    circle("j.0", radius: 3pt, fill: red)
    intersections("k", "lright", "lhbot")
    circle("k.0", radius: 3pt, fill: red)
    intersections("l", "lleft", "lhbot")
    circle("l.0", radius: 3pt, fill: red)
    line("i.0", "j.0", "k.0", "l.0", close: true)

    // line((0, H), (rel: (angle: -30deg, radius: 10)))
    line((0, H), (rel: (H, -(x2 - x1))), name: "houghline", stroke: 0pt)
    line((0, H), (project: (), onto: ("lleft.start", "lleft.end")),
        stroke: red, label: [world], name: "projectionline")
    content(("projectionline.start", 60%, "projectionline.end"), anchor: "south", angle: "projectionline.end", padding: 0.1em)[
        #set text(size: 1.5em)
        $r$
    ]
    cetz.angle.angle((0, H), (1, H), "projectionline.end", direction: "ccw", radius: 1, mark: (end: ">"), label: [
        #set text(size: 1.5em)
        $theta$
    ])

})
