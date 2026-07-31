import paho.mqtt.client as mqtt
import time
import random
from dataclasses import dataclass
from collections import namedtuple
from utils import is_running_on_raspberry_pi

@dataclass(frozen=True)
class Slot:
    id: int
    col: int
    row: int
    colspan: int = 1
    rowspan: int = 1
    kind: str = "standard"   # "standard" | "featured" — seam for future repeat-fill dispatch

SlotGeometry = namedtuple("SlotGeometry", ["id", "kind", "x", "y", "width", "height"])

class StateEngine:
    # Hardcoded physical grid layout: one 2x2 "featured" slot (bottom-right) + 11 standard slots.
    #  1,  2,  3,  4,  5
    #  6,  -,  8,  9, 10
    #  -,  -, 13, 14, 15
    SLOT_LAYOUT = [
        Slot(id=1,  col=0, row=0),
        Slot(id=2,  col=1, row=0),
        Slot(id=3,  col=2, row=0),
        Slot(id=4,  col=3, row=0),
        Slot(id=5,  col=4, row=0),
        Slot(id=6,  col=0, row=1, colspan=2, rowspan=2, kind="featured"),
        Slot(id=8,  col=2, row=1),
        Slot(id=9,  col=3, row=1),
        Slot(id=10, col=4, row=1),
        Slot(id=13, col=2, row=2),
        Slot(id=14, col=3, row=2),
        Slot(id=15, col=4, row=2),
    ]

    CELL_FILL_RATIO = 0.93   # tuned so standard slots render at the same visual size as before
    DEFAULT_TARGET_SIZE = 800

    # Global drawing style for all plotted portraits (None = classic contour tracing).
    # '+'-combinable tokens: 'features', 'outline', 'shade', 'hair', 'landmarks', 'oneline',
    # e.g. "features+outline+shade+oneline". Test styles with parsefile.py first.
    DRAWING_STYLE = "features+hair+outline+shade+oneline"
    FEATURE_RADIUS = 18   # features: max distance (px at 800) from a landmark line
    SHADES = 2            # shade: tone count incl. paper white
    HATCH_SPACING = 10    # shade: hatch line spacing (px at 800)
    SIMPLIFY = 80         # RDP simplification strength in percent (higher = fewer points)
    HAIR_STROKES = 30     # hair: target brush stroke count

    @classmethod
    def _standard_shuffle_blocks(cls):
        """Derives shuffle blocks from SLOT_LAYOUT instead of hand-maintaining a separate list:
        one block per row of "standard" slots, ordered by column. Excludes the featured slot(s),
        so repositioning/resizing the featured slot in SLOT_LAYOUT keeps this in sync automatically."""
        rows = {}
        for slot in cls.SLOT_LAYOUT:
            if slot.kind == "standard":
                rows.setdefault(slot.row, []).append(slot)
        return [
            [slot.id for slot in sorted(rows[row], key=lambda s: s.col)]
            for row in sorted(rows)
        ]

    @classmethod
    def _featured_slot_id(cls):
        return next(s.id for s in cls.SLOT_LAYOUT if s.kind == "featured")

    def __init__(self):
        # State
        self.state = "Startup"
        self.currentPhotoPath = ""
        self.currentWorkPath = ""
        self.currentSVGPath = ""
        self.imagesPerRow = 5
        self.imagesPerColumn = 3
        self.slots = {s.id: s for s in self.SLOT_LAYOUT}
        self.slot_ids = [s.id for s in self.SLOT_LAYOUT]  # all 12 physical slots, incl. featured
        self.featured_slot_id = self._featured_slot_id()
        self.paperSizeX = 1587 #1191 multiplied by higher 96 dpi of Nextdraw
        self.paperSizeY = 1122 #841 multiplied by higher 96 dpi of Nextdraw
        self.workID = 0
        self.photoID = []  # populated by reset_photo_id() below (excludes the featured slot)
        self.reset_timeout_s = 15
        self.last_update_time = 0
        self.stresslevel = 0.0
        self.last_draw_end_time = None
        self.next_draw_start_time = None
        self.min_stress_time = 30     # seconds (max stress)
        self.max_stress_time = 300    # seconds (min stress, 5 min)
        self.stress_decay_rate = 0.0025
        self.update_interval = 1
        self.transitions = {
            "Startup": ["Waiting", "ResetPending", "Test"],
            "Waiting": ["Tracking"],
            "Tracking": ["Working", "Snapping", "Tracking"],
            "Working": ["Tracking"],
            "Snapping": ["Tracking", "Processing"],
            "Processing": ["Drawing", "Waiting"],
            "Drawing": ["Redrawing", "Waiting", "ResetPending"],
            "Redrawing": ["Waiting", "ResetPending"],
            "ResetPending": ["Waiting", "Template"],
            "Template": ["ResetPending"],
            "Test": ["Waiting", "Drawing"]
        }

        if is_running_on_raspberry_pi():
            print(f"\033[1;33m📱 Raspberry Pi found: Connecting to broker\033[0m.")
            self.broker_address = "localhost"
            self.client = mqtt.Client("StateEngine_Client")
            self.client.on_connect = self.on_connect
            self.client.connect(self.broker_address)
            self.client.subscribe("lcd/buttons")
            self.client.loop_start()
            self.client.on_message = self.on_message
            print("MQTT broker started.")
        else:
            print(f"\033[1;32m🖥️ Raspberry Pi not found: Entering test mode\033[0m.")
            self.state = "Test"
        
        print("Starting StateEngine ...")
        self.reset_photo_id()  # Shuffle the photoID list on startup
    
    # State
    # ------------------------------------------------------------------------     
    def get_state(self):
        return self.state
    
    def change_state(self, new_state):
        # Only update state if transition is possible
        if new_state in self.transitions[self.state]:
            print(f"State change from {self.state} to {new_state}.")
            self.state = new_state
            if self.state == "Drawing":
                # Special case to display the current drawing image 
                message = f"{new_state}-{self.photoID[-1]}"  # Display the current last photoID
                self.publish_message("state_engine/state", message)
            else:
                self.publish_message("state_engine/state", new_state)
        else:
            print(f"Invalid transition from {self.state} to {new_state}.")

    def update_image_path(self, photo_path):
        self.currentPhotoPath = photo_path
        print(f"Photo path updated to {photo_path}.")
        self.publish_message("state_engine/photo_path", photo_path)
            
    def update_photo_id(self):
        if self.photoID:
            removed_id = self.photoID.pop()  # Remove the last photoID
            print(f"Photo ID removed: {removed_id}, remaining IDs: {self.photoID}")
            if not self.photoID:  # If the list becomes empty, trigger reset
                print("All images processed. Transitioning to ResetPending.")
                self.change_state("ResetPending")
        else:
            print("No photo IDs available.")

    def reset_photo_id(self):
        blocks = self._standard_shuffle_blocks()
        for block in blocks:
            random.shuffle(block)
        self.photoID = [item for block in blocks for item in block]
        print(f"Photo IDs reset and shuffled within blocks: {self.photoID}")
    
    def update_work_id(self):
        self.workID += 1
        print(f"Work ID: {self.workID}")
        
    def reset_work_id(self):
        print(f"Reset Work ID: {self.workID} -> 0")
        self.workID = 0
      
    def _base_cell_dims(self):
        """Force square cells: derive one shared cellSize from whichever axis is tighter,
        then center the resulting (smaller-than-paper on the slack axis) grid within the paper."""
        borderSize, gutterSize = 50, 50
        maxX = self.paperSizeX - (borderSize * 2)
        maxY = self.paperSizeY - (borderSize * 2)

        candidateWidth  = (maxX - gutterSize * (self.imagesPerRow - 1)) / self.imagesPerRow
        candidateHeight = (maxY - gutterSize * (self.imagesPerColumn - 1)) / self.imagesPerColumn
        cellSize = min(candidateWidth, candidateHeight)

        gridWidth  = cellSize * self.imagesPerRow    + gutterSize * (self.imagesPerRow - 1)
        gridHeight = cellSize * self.imagesPerColumn + gutterSize * (self.imagesPerColumn - 1)

        borderX = (self.paperSizeX - gridWidth) / 2
        borderY = (self.paperSizeY - gridHeight) / 2

        return cellSize, cellSize, borderX, borderY, gutterSize

    def get_slot_geometry(self, slot_id):
        """Returns the SlotGeometry (origin + size) for a given slot id, accounting for spans."""
        slot = self.slots[slot_id]
        cellWidth, cellHeight, borderX, borderY, gutterSize = self._base_cell_dims()

        x = borderX + slot.col * (cellWidth + gutterSize)
        y = borderY + slot.row * (cellHeight + gutterSize)
        width  = cellWidth  * slot.colspan + gutterSize * (slot.colspan - 1)
        height = cellHeight * slot.rowspan + gutterSize * (slot.rowspan - 1)

        return SlotGeometry(id=slot.id, kind=slot.kind, x=x, y=y, width=width, height=height)

    def compute_scale_factor(self, cell_width, cell_height, source_size, fill_ratio=None):
        """Derives create_output_svg's scale_factor from actual cell size instead of a fixed constant.
        fill_ratio defaults to CELL_FILL_RATIO (leaves a margin for real artwork); pass 1.0 for
        alignment/debug guides that should trace the true cell boundary exactly."""
        if fill_ratio is None:
            fill_ratio = self.CELL_FILL_RATIO
        return (min(cell_width, cell_height) / source_size) * fill_ratio
    
    # Stresslevel
    # ------------------------------------------------------------------------    
    def calculate_stresslevel(self, interval_seconds):
        """
        Compute stress level based on the time between drawings.
        Shorter intervals → higher stress.
        Long intervals → stress decays toward 0.0.
        """
        min_time = self.min_stress_time
        max_time = self.max_stress_time

        # Clamp interval to min/max for stress scaling
        clamped = max(min_time, min(interval_seconds, max_time))
        stress = 1.0 - (clamped - min_time) / (max_time - min_time)
        alpha = 0.3
        smoothed_stress = (alpha * stress) + (1 - alpha) * self.stresslevel

        # Idle decay: reduce stress if interval is long
        idle_decay = self.stress_decay_rate * interval_seconds
        smoothed_stress = max(0.0, smoothed_stress - idle_decay)

        self.stresslevel = smoothed_stress

        print(f"⏱️ Interval: {interval_seconds:.1f}s → StressLevel: {self.stresslevel:.2f}")
        return self.stresslevel

    def update_stresslevel_from_interval(self):
        """
        Compute and update stresslevel based on time since the last drawing ended.
        If no previous drawing time exists, the stresslevel remains unchanged.
        Returns the current stresslevel.
        """
        if not self.last_draw_end_time:
            print("No previous drawing found. Using default stresslevel.")
            return self.stresslevel

        interval = time.time() - self.last_draw_end_time
        return self.calculate_stresslevel(interval)
    
    def get_stress_scaled_params(self):
        s = max(0.0, min(1.0, self.stresslevel))
        min_paths = int(40 - (40 - 20) * s)
        max_paths = int(140 - (140 - 80) * s)
        min_contour_area = int(10 + (20 - 10) * s)
        return {"min_paths": min_paths, "max_paths": max_paths, "min_contour_area": min_contour_area}

    def get_render_params(self, slot_id):
        """Composes stress-scaled trace params with slot-size scaling, so a bigger (spanning)
        slot gets proportionately more tracing resolution/density instead of a stretched, sparse trace."""
        slot = self.slots[slot_id]
        cellWidth, cellHeight, _, _, gutterSize = self._base_cell_dims()
        slot_width  = cellWidth  * slot.colspan + gutterSize * (slot.colspan - 1)
        slot_height = cellHeight * slot.rowspan + gutterSize * (slot.rowspan - 1)
        area_ratio = (slot_width * slot_height) / (cellWidth * cellHeight)
        linear_ratio = area_ratio ** 0.5

        base = self.get_stress_scaled_params()
        return {
            "target_width":     round(self.DEFAULT_TARGET_SIZE * linear_ratio),
            "target_height":    round(self.DEFAULT_TARGET_SIZE * linear_ratio),
            "max_paths":        round(base["max_paths"] * area_ratio),
            "min_contour_area": round(base["min_contour_area"] * area_ratio),
            "min_paths":        base["min_paths"],
            "style":            self.DRAWING_STYLE,
            "feature_radius":   self.FEATURE_RADIUS,
            "shades":           self.SHADES,
            "hatch_spacing":    self.HATCH_SPACING,
            "simplify":         self.SIMPLIFY,
            "hair_strokes":     self.HAIR_STROKES,
            "stress":           max(0.0, min(1.0, self.stresslevel)),
        }

    # Messages
    # ------------------------------------------------------------------------
    def on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            print("Connected to MQTT broker")
        else:
            print(f"Failed to connect to MQTT broker with error code {rc}")
            
    def on_message(self, client, userdata, msg): 
        # Manually handle reset
        message = msg.payload.decode()
        reset_keys = ["KEY2", "KEY3", "UP", "DOWN", "LEFT", "RIGHT"]
        template_keys = ["KEY1"]
        redraw_keys = ["KEY2"]
        more_details_keys = ["UP"]
        less_details_keys = ["DOWN"]
        
        # Handle messages received on subscribed topics
        print(f"Received message on topic '{msg.topic}': {message}")
        current_time = time.time()

        if self.state == "ResetPending":
            if message in reset_keys:
                print("Reset confirmed")
                self.reset_photo_id()  # Reset and shuffle photo IDs
                self.change_state("Waiting")
                time.sleep(1)    
            elif message in template_keys:
                print("Applying template")
                self.change_state("Template")
                time.sleep(1)
        
        # Handler for each state
        elif self.state == "Waiting":
            self.reset_work_id()
            self.change_state("Tracking")
                
        elif self.state == "Working":
            self.change_state("Tracking")
            
        elif self.state == "Drawing":
            if message in redraw_keys:
                print("Redraw triggered")
                self.change_state("Redrawing")
                time.sleep(1)

        elif self.state == "Redrawing":
            # Featured-slot reprint already in progress — ignore repeat KEY2 presses until
            # it finishes and a fresh "Drawing" state re-arms the redraw trigger.
            pass

        else:
            # print(f"Unexpected state: {self.state}") 
            pass
        
    def publish_message(self, topic, message):
        self.client.publish(topic, message)
        

