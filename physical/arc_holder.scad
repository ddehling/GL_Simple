// Arc holder with numbered holding positions
// 4-foot radius, 45-degree arc, 3cm tall, 5cm thick

// --- Parameters ---
arc_radius     = 4 * 304.8;    // 4 feet inner radius in mm (1219.2mm)
arc_angle      = 45 / 4;       // arc sweep in degrees (11.25°)
arc_height     = 35;           // 3.5cm tall
arc_thickness  = 65;           // 6.5cm thick (radial) - wide enough so holes clear channel
hole_spacing   = 30;           // 3cm between holes along outer edge
holerad        = 3.75;         // torus hole radius (matches original)
torus_offset   = 15;           // torus ring offset (matches original)
text_size      = 8;            // label size

// --- Derived ---
outer_radius    = arc_radius + arc_thickness;
outer_arc_len   = outer_radius * arc_angle * PI / 180;
num_holes       = 8;            // fixed 8 holes
// 7 gaps between holes (1 unit each) + 2 half-unit margins = 8 units total
angle_per_hole  = arc_angle / (num_holes - 1 + 2 * 0.5);  // 1 unit spacing
start_offset    = angle_per_hole / 2;            // half unit from edge

echo(str("Outer arc length: ", outer_arc_len, " mm"));
echo(str("Number of hole positions: ", num_holes));
echo(str("Angular spacing: ", angle_per_hole, " deg"));

// --- Pie/sector module ---
module sector_2d(radius, angle) {
    // Creates a 2D sector from 0 to angle degrees
    intersection() {
        circle(r = radius, $fn = 360);
        points = concat(
            [[0, 0]],
            [for (a = [0 : 1 : angle]) [(radius + 1) * cos(a), (radius + 1) * sin(a)]],
            [[(radius + 1) * cos(angle), (radius + 1) * sin(angle)]]
        );
        polygon(points);
    }
}

// --- Arc module with sequential numbering ---
// num_offset: starting number for labels (0-based: first label = num_offset + 1)
module arc_piece(num_offset = 0) {
    // How far (in angle) from each end to trim the inner wall
    inner_trim_len = 30;         // 3cm in mm
    inner_trim_angle = (inner_trim_len / arc_radius) * 180 / PI;

    // Center the arc at the origin by translating the arc center back
    translate([-arc_radius - arc_thickness / 2, 0, 0])
    difference() {

        // ===== Solid arc =====
        linear_extrude(arc_height)
        rotate([0, 0, -arc_angle / 2])
        difference() {
            sector_2d(outer_radius, arc_angle);
            circle(r = arc_radius, $fn = 360);
        }

        // ===== Trim inner wall at both ends (first & last 3cm) =====
        trim_r = arc_radius + arc_thickness / 2 - 25.4 / 2 - 0.1;
        overcut = 0.5;  // degrees of overcut for clean boolean
        inner_trim_angle = (inner_trim_len / arc_radius) * 180 / PI;

        // Use rotate_extrude with limited angle for precise arc-shaped cuts
        // Start end
        translate([0, 0, -0.5])
        rotate([0, 0, -arc_angle / 2 - overcut])
        rotate_extrude(angle = inner_trim_angle + 2 * overcut, $fn = 360)
        translate([arc_radius - 1, 0, 0])
        square([trim_r - arc_radius + 2, arc_height + 2]);

        // End end (identical shape, rotated to other end)
        translate([0, 0, -0.5])
        rotate([0, 0, arc_angle / 2 - inner_trim_angle - overcut])
        rotate_extrude(angle = inner_trim_angle + 2 * overcut, $fn = 360)
        translate([arc_radius - 1, 0, 0])
        square([trim_r - arc_radius + 2, arc_height + 2]);

        // ===== Holding holes (torus cutouts on outer face) =====
        for (i = [0 : num_holes - 1]) {
            ang = -arc_angle / 2 + start_offset + i * angle_per_hole;

            translate([outer_radius * cos(ang),
                       outer_radius * sin(ang),
                       arc_height])
            rotate([0, 0, ang])
            rotate([90, 0, 0])
            rotate_extrude(convexity = 10, $fn = 50)
                translate([torus_offset, 0, 0])
                    circle(r = holerad, $fn = 20);
        }

        // ===== Number labels (engraved into top surface, odd holes only) =====
        for (i = [0 : 2 : num_holes - 1]) {
            ang = -arc_angle / 2 + start_offset + i * angle_per_hole;
            label_r = arc_radius + arc_thickness * 0.5;
            label_num = num_offset + floor(i / 2) + 1;

            translate([label_r * cos(ang),
                       label_r * sin(ang),
                       arc_height - 1.5])
            rotate([0, 0, ang - 90])
            linear_extrude(3)
            text(str(label_num), size = text_size, halign = "center", valign = "center");
        }

        // ===== Number labels (engraved into outer face, odd holes only) =====
        for (i = [0 : 2 : num_holes - 1]) {
            ang = -arc_angle / 2 + start_offset + i * angle_per_hole;
            label_num = num_offset + floor(i / 2) + 1;

            translate([(outer_radius + 1) * cos(ang),
                       (outer_radius + 1) * sin(ang),
                       text_size * 0.6])
            rotate([0, 0, ang])
            rotate([0, 90, 0])
            rotate([0, 0, 90])
            translate([0, 0, -5])
            linear_extrude(6)
            text(str(label_num), size = text_size, halign = "center", valign = "center");
        }

        // ===== Number labels (engraved into bottom surface, odd holes only) =====
        for (i = [0 : 2 : num_holes - 1]) {
            ang = -arc_angle / 2 + start_offset + i * angle_per_hole;
            label_r = arc_radius + arc_thickness * 0.5 + channel_size / 2 + (outer_radius - (arc_radius + arc_thickness / 2 + channel_size / 2)) / 2;
            label_num = num_offset + floor(i / 2) + 1;

            translate([label_r * cos(ang),
                       label_r * sin(ang),
                       1.5])
            rotate([0, 0, ang - 90])
            rotate([180, 0, 0])
            linear_extrude(3)
            text(str(label_num), size = text_size, halign = "center", valign = "center");
        }

        // ===== 1-inch square channel through middle of arc, sitting on z=0 =====
        channel_size = 25.4;
        channel_radius = arc_radius + arc_thickness / 2;

        rotate([0, 0, -arc_angle / 2 - 1])
        rotate_extrude(angle = arc_angle + 2, $fn = 360)
        translate([channel_radius - channel_size / 2, 0, 0])
        square([channel_size, channel_size], center = false);
    }
}

// --- Number of labeled holes per arc piece (odd holes only) ---
labels_per_piece = ceil(num_holes / 2);

// --- Spacing between copies along X ---
copy_spacing = outer_arc_len + 50;  // arc length + 50mm gap

// --- 3 copies spaced along X, with sequential numbering ---
for (c = [0 : 2]) {
    translate([c * copy_spacing, 0, 0])
    arc_piece(num_offset = c * labels_per_piece);
}
