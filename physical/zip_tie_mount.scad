/*
 * Box-to-Pipe Zip Tie Mount
 *
 * Flat base glues to a plastic box.
 * Torus ring (rotate-extruded circle) accepts zip tie for pipe attachment.
 * Ring is embedded in the base up to its center — smooth, no pinch points.
 * Cradle on top of ring seats a 1" EMT conduit running along the ring plane.
 *
 * Print base-down, no supports needed.
 */

$fn = 100;

/* ══════ Base Plate ══════ */
base_dia    = 75;       // mm diameter, ~2 inches
base_h      = 10;        // mm thickness

/* ══════ Ring (Torus) ══════ */
ring_R      = 20;       // major radius (torus center to tube center)
ring_r      = 15;       // minor radius (tube cross-section) — ~2 cm diameter
                        // Inner opening diameter = 2*(ring_R - ring_r) = 10 mm

/* ══════ Pipe Cradle ══════ */
emt_od       = 29.5;    // mm, 1" EMT outer diameter
cradle_depth = 20;       // mm, how deep pipe nests into ring top

/* ══════ Derived ══════ */
z_top = base_h + ring_R + ring_r;   // top of torus

/* ══════ Modules ══════ */

module base_plate() {
    cylinder(d = base_dia, h = base_h);
}

// Torus standing vertically, center at base surface height
module torus_ring() {
    translate([0, 0, base_h])
        rotate([90, 0, 0])
            rotate_extrude()
                translate([ring_R, 0])
                    circle(r = ring_r);
}

// Pipe-shaped cutout along X axis at top of ring
module pipe_cradle() {
    translate([0, 0, z_top + emt_od/2 - cradle_depth])
        rotate([0, 90, 0])
            cylinder(d = emt_od, h = base_dia * 2, center = true);
}

/* ══════ Assembly ══════ */

intersection() {
    difference() {
        union() {
            base_plate();
            torus_ring();
        }
        // Cut cradle into top of ring
        pipe_cradle();
    }
    // Keep only z >= 0
    translate([0, 0, 500])
        cube(1000, center = true);
}
