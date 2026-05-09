// =============================================================
// Paracord Cinch Fastener — 3-peg friction lock
// =============================================================
//
// Cylinder body, flat bottom, axial through-hole, zip-tie groove,
// three mooring-pillar pegs, plus:
//
//   - OUTER grooves (az 90°, 270°): full-length retention
//     channels on the cylinder sides, where the radial pegs
//     emerge. Run all the way through both Y faces.
//
//   - INNER grooves (az 160°, 200°): two trenches on the +Z
//     surface that run from the +Y face down and terminate on
//     the ring around peg C. Cord can travel through either
//     trench into the ring.
//
//   - RING around peg C: a full 360° trench on the +Z surface
//     centered on peg C. Cord can enter from either inner
//     trench and wrap peg C any number of times.
//
//   - PEGS:
//       A — radial, tilts -peg_tilt_factor·90° from +Z (sits
//           between inner top groove az 200° and outer side
//           groove az 270°), at peg_y_top
//       B — radial, tilts +peg_tilt_factor·90° from +Z (sits
//           between inner top groove az 160° and outer side
//           groove az 90°), at peg_y_top
//       C — vertical on top, in the middle of the U-bend at
//           peg_y_bot
//
// =============================================================

$fn = 64;
$fn_round = 24;

// === PARAMETERS ===

// Cylinder body
R         = 11;
body_l    = 50;
h_flat    = 9;

// Through-hole
hole_d    = 5.0;

// Cord grooves
ch_w      = 4.4;
ch_d      = 3.0;

// Zip-tie groove
zt_w      = 3.8;
zt_d      = 1.5;
zt_y      = 0;

// Outer groove azimuths (full-length retention channels on
// the sides of the cylinder, where pegs A and B emerge)
groove_az_1 = 90;
groove_az_2 = 270;
// Inner groove azimuths (U-channel legs on top)
inner_groove_az_1 = 160;
inner_groove_az_2 = 200;

// Mooring pegs (shorter and wider than v1 — stubby aspect ratio
// for print strength, esp. for the radial pegs which take cord
// load in cantilever)
peg_shaft_d = 8;
peg_cap_d   = 12;
peg_cap_h   = 2.5;
peg_height  = 6;            // height above the local body surface
// Y of the radial peg pair — pushed out toward the +Y face so
// the peg shafts sit flush against it (peg outer edge at face).
peg_y_top   = body_l/2 - peg_shaft_d/2;
peg_y_bot   = -12;          // Y of peg C (vertical, top center)
peg_clearance = 1;
// Radial pegs sit BETWEEN the outer side grooves (az 90°/270°)
// and the inner top grooves (az 160°/200°). Tilt factor is the
// fraction of 90°: 0.6 → 54° tilt → peg az 126°/234° (midpoint
// of the two groove gaps).
peg_tilt_factor = 0.6;

// Ring channel around peg C — a full 360° trench centered on
// the peg, so the cord can wrap the peg from any direction.
// Channel inner edge sits AT the peg surface (ubend_radius
// = peg radius + half channel width) so the cord rides directly
// against the peg shaft when it wraps.
ubend_radius = peg_shaft_d/2 + ch_w/2;

// Smoothing
edge_fillet = 0.6;

// === DERIVED ===
// Inner slots run all the way through the +Y face (past pegs A
// and B) so the cord exits the trench right at the pegs and
// wraps them. The pegs sit in the trench path; their shafts get
// notched where the trench cuts through, but the visible peg
// above the cylinder surface is intact.
inner_slot_y_max = body_l/2 + 1;
// On the -Y side the inner slots end where they meet the ring
// around peg C. Endpoints lie on a circle of radius ubend_radius
// centered on peg C; for the endpoint X to match the inner-slot
// center X, the endpoint Y is solved from
// x² + (y - peg_y_bot)² = r² .
inner_slot_x        = (R - ch_d/2 + 1) * sin(180 - inner_groove_az_1);
inner_slot_y_min    = peg_y_bot
                    + sqrt(max(0, ubend_radius*ubend_radius
                                 - inner_slot_x*inner_slot_x));

// === HELPERS ===

module rounded_cube(size, r) {
    minkowski() {
        cube([max(0.01, size[0] - 2*r),
              max(0.01, size[1] - 2*r),
              max(0.01, size[2] - 2*r)], center = true);
        sphere(r = r, $fn = $fn_round);
    }
}

// === BODY AND CUTS ===

module body_solid() {
    difference() {
        rotate([90, 0, 0])
            cylinder(r = R, h = body_l, center = true);
        translate([-(R + 1), -(body_l/2 + 1), -2*R])
            cube([2*R + 2, body_l + 2, 2*R - h_flat]);
    }
}

module through_hole() {
    rotate([90, 0, 0])
        cylinder(d = hole_d, h = body_l + 4, center = true);
}

module top_groove(azimuth) {
    // Outer side slot. Runs the full body length. Cross-section
    // is a pentagon — rectangle with the local -X face peaked
    // into a 45° gable. After this module's [0, 180-azimuth, 0]
    // rotation, the local -X face becomes the world +Z face for
    // the side slots (azimuth = 90° / 270°), so the slot ceiling
    // is self-supporting when printed flat-bottom-down.
    cut_y_min    = -body_l/2 - 1;
    cut_y_max    =  body_l/2 + 1;
    cut_y_center = (cut_y_min + cut_y_max) / 2;
    cut_y_size   = cut_y_max - cut_y_min;

    rotate([0, 180 - azimuth, 0])
        translate([0, cut_y_center, R - ch_d/2 + 1])
            chamfered_slot(ch_w, cut_y_size, ch_d + 2);
}

module chamfered_slot(width, length, depth) {
    // Hexagon-cross-section slot extruded along Y. Cross-section
    // in local X-Z is a rectangle with all four corners chamfered
    // at 45° (chamfer size = width/2), so the top and bottom
    // edges collapse to peaks at (0, ±depth/2) and the local ±X
    // faces become short flat segments at lz ∈ [-depth/2 + a,
    // depth/2 - a]. This is symmetric for both ±X-facing rotations
    // (az 90° and 270°): in either case the world +Z ceiling
    // becomes a small flat with 45° slopes on each side — bridge-
    // able on FDM.
    a = width/2;
    rotate([90, 0, 0])
        linear_extrude(height = length, center = true)
            polygon([
                [-width/2 + a, +depth/2],   // collapsed top peak
                [+width/2,     +depth/2 - a],
                [+width/2,     -depth/2 + a],
                [+width/2 - a, -depth/2],   // collapsed bottom peak
                [-width/2,     -depth/2 + a],
                [-width/2,     +depth/2 - a]
            ]);
}

module inner_top_groove(azimuth) {
    // Inner U-channel leg. -Y end connects to the U-bend, +Y end
    // stops short of the +Y pegs.
    cut_y_min    = inner_slot_y_min;
    cut_y_max    = inner_slot_y_max;
    cut_y_center = (cut_y_min + cut_y_max) / 2;
    cut_y_size   = cut_y_max - cut_y_min;

    rotate([0, 180 - azimuth, 0])
        translate([0, cut_y_center, R - ch_d/2 + 1])
            rounded_cube([ch_w, cut_y_size, ch_d + 2], edge_fillet);
}

module ring_around_peg_c() {
    // Full 360° trench on the cylinder's +Z surface, centered
    // on peg C. The two inner trenches feed into this ring at
    // (±inner_slot_x, inner_slot_y_min) — points on the ring's
    // circumference. Cord can enter the ring from either trench
    // and wrap peg C any number of times.
    translate([0, peg_y_bot, 0])
        rotate_extrude(angle = 360, $fn = 192)
            translate([ubend_radius - ch_w/2,
                       R - ch_d/2 + 1 - (ch_d + 2)/2])
                square([ch_w, ch_d + 2]);
}

module peg_shape() {
    // Generic mooring-pillar shape, oriented along +Z. Used by
    // both vertical and radial pegs (the radial ones get rotated).
    //
    // Layered from bottom up:
    //   • shaft cylinder at peg_shaft_d
    //   • 45° conical taper from peg_shaft_d → peg_cap_d (replaces
    //     the original abrupt step — self-supporting on FDM)
    //   • short flat disc at peg_cap_d filling any remaining cap
    //     height (peg_cap_h - cone height)
    //   • flattened half-sphere dome at peg_cap_d on top
    cone_h  = (peg_cap_d - peg_shaft_d) / 2;   // 45° ⇒ cone_h = radial growth
    flat_h  = max(0, peg_cap_h - cone_h);
    shaft_h = R + peg_height - peg_cap_h;

    cylinder(d = peg_shaft_d, h = shaft_h, $fn = 48);
    translate([0, 0, shaft_h]) {
        cylinder(d1 = peg_shaft_d, d2 = peg_cap_d,
                 h = cone_h, $fn = 48);
        translate([0, 0, cone_h])
            cylinder(d = peg_cap_d, h = flat_h, $fn = 48);
        translate([0, 0, cone_h + flat_h])
            scale([1, 1, 0.5])
                sphere(d = peg_cap_d, $fn = 48);
    }
}

module mooring_peg_vertical(x, y) {
    translate([x, y, 0])
        peg_shape();
}

module mooring_peg_radial(side_factor, y) {
    // Radial peg sticking out from the cylinder side at a tilt.
    // side_factor is multiplied by 90° to get the tilt angle from
    // the +Z (top) axis: ±1.0 → ±90° (dead on +X / -X equator),
    // ±0.6 → ±54° (between the inner top groove and outer side
    // groove on each side).
    rotate([0, side_factor * 90, 0])
        translate([0, y, 0])
            peg_shape();
}

module peg_support_wedge(support_l, support_h, thickness, sign_y, dive) {
    // Wedge gusset built as the convex hull of an apex line
    // (above body surface) and a rectangular root box (below
    // body surface). Cross-section in X-Z at any Y is a TRIANGLE
    // peaked at peg-local X=0 — cord wrapping the peg can ride
    // over the X=0 ridge rather than catching on a flat top.
    //
    //  peg apex (0, 0, R+H) ─╮
    //                          ╲     ridge sloping down to nub
    //   nub apex (0, L, R) ─────╯    ← body surface
    //
    //   base   (±t/2, 0..L, R-dive)  ← root, inside body
    //
    // Peak height tapers linearly from H above body surface at
    // Y=0 (peg side) to 0 at Y=L (nub side). Root depth `dive`
    // gives solid fusion with the cylinder body in CSG union.
    L = sign_y * support_l;
    H = support_h;
    t = thickness;
    d = dive;

    hull() {
        translate([0,    0, R + H])  cube(0.001, center=true);  // peg apex
        translate([0,    L, R    ])  cube(0.001, center=true);  // nub apex
        translate([-t/2, 0, R - d])  cube(0.001, center=true);  // root corners
        translate([+t/2, 0, R - d])  cube(0.001, center=true);
        translate([+t/2, L, R - d])  cube(0.001, center=true);
        translate([-t/2, L, R - d])  cube(0.001, center=true);
    }
}

module zt_groove() {
    translate([0, zt_y, 0])
        difference() {
            rotate([90, 0, 0])
                cylinder(r = R + 1, h = zt_w, center = true);
            rotate([90, 0, 0])
                cylinder(r = R - zt_d, h = zt_w + 2, center = true);
        }
}

// === BUILD ===
// All pegs and their gussets are wrapped in a 2 mm world -Z
// translate (everything lowered by 2 mm). Body and cutouts stay
// at their original Z.
union() {
    difference() {
        union() {
            body_solid();
            translate([0, 0, -2]) {
                mooring_peg_radial(-peg_tilt_factor, peg_y_top);  // peg A
                mooring_peg_radial(+peg_tilt_factor, peg_y_top);  // peg B
                mooring_peg_vertical(0, peg_y_bot);               // peg C

                // Gussets for pegs A & B
                rotate([0, -peg_tilt_factor * 90, 0])
                    translate([0, peg_y_top, 0])
                        peg_support_wedge(9, 9, peg_shaft_d, -1, 2);
                rotate([0, +peg_tilt_factor * 90, 0])
                    translate([0, peg_y_top, 0])
                        peg_support_wedge(9, 9, peg_shaft_d, -1, 2);
            }
        }
        through_hole();
        top_groove(groove_az_1);              // outer, full length
        top_groove(groove_az_2);              // outer, full length
        inner_top_groove(inner_groove_az_1);  // U leg
        inner_top_groove(inner_groove_az_2);  // U leg
        ring_around_peg_c();                  // full ring around peg C
        zt_groove();
    }

    // Peg C gusset added AFTER the difference so the ring trench
    // around peg C doesn't carve through it. Also lowered 2 mm.
    translate([0, 0, -2])
        translate([0, peg_y_bot, 0])
            peg_support_wedge(
                (zt_y - zt_w/2 - 1) - peg_y_bot,  // length to nub
                peg_height,                        // height
                peg_shaft_d,                       // thickness
                +1,                                // +Y direction
                5                                  // dive
            );
}
