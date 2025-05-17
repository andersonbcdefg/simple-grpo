from PIL import Image, ImageDraw
import math
import random
import re

class TimeObj:
    def __init__(self, hours=None, minutes=None, seconds=None):
        if hours is None and minutes is None and seconds is None:
            # Generate random time ensuring hours are 1-12 for typical analog display
            self.hours = random.randint(1, 12)
            self.minutes = random.randint(0, 59)
            self.seconds = random.randint(0, 59)
        else:
            # Ensure hours are within a valid range if provided
            self.hours = hours if hours is not None else 0
            self.minutes = minutes if minutes is not None else 0
            self.seconds = seconds if seconds is not None else 0
            # Basic validation/wrapping could be added here if needed

    def subtract(self, other):
        """ Calculates the absolute difference in seconds between two TimeObj instances. """
        # Convert both times to seconds from 12:00:00 for comparison
        # Handle 12 o'clock properly (treat as 0 for calculation if needed, or adjust)
        # Using a 12-hour cycle, the difference should consider the circular nature.
        # E.g., 1:00:00 vs 11:00:00 is 2 hours diff, not 10.

        # Convert to total seconds within a 12-hour cycle (12 * 3600 = 43200 seconds)
        total_seconds_self = ((self.hours % 12) * 3600 + self.minutes * 60 + self.seconds)
        total_seconds_other = ((other.hours % 12) * 3600 + other.minutes * 60 + other.seconds)

        difference_in_seconds = abs(total_seconds_self - total_seconds_other)

        # Account for wrapping around the 12-hour mark
        # The maximum difference on a 12-hour clock is 6 hours (21600 seconds)
        if difference_in_seconds > 21600:
            difference_in_seconds = 43200 - difference_in_seconds

        hours = difference_in_seconds // 3600
        minutes = (difference_in_seconds % 3600) // 60
        seconds = difference_in_seconds % 60
        
        return difference_in_seconds, {
            "hours": hours,
            "minutes": minutes,
            "seconds": seconds,
            "formatted": f"{hours:02}:{minutes:02}:{seconds:02}"
        }

    def __str__(self):
        # Standard format HH:MM:SS, padding with zeros
        return f"[{self.hours:02}:{self.minutes:02}:{self.seconds:02}]"

    @classmethod
    def from_string(cls, time_str):
        """ Creates a TimeObj from a string like [HH:MM:SS] or HH:MM:SS. """
        match = re.match(r"\[?(\d{1,2}):(\d{2}):(\d{2})\]?", time_str)
        if match:
            h, m, s = map(int, match.groups())
            # Basic validation (e.g., hours 1-12, min/sec 0-59)
            if 1 <= h <= 12 and 0 <= m <= 59 and 0 <= s <= 59:
                return cls(hours=h, minutes=m, seconds=s)
        return None # Return None if format is invalid


class ClockGen:
    def __init__(self, time_obj):
        if not isinstance(time_obj, TimeObj):
            raise ValueError("ClockGen requires a TimeObj instance.")
        self.time_obj = time_obj

    def generate_clock(self, filename="clock.png"):
        # Define image properties
        img_size = 224 # Standard size for many vision models
        center = (img_size // 2, img_size // 2)
        radius = img_size // 2 - 10 # Adjusted padding
        bg_color = "white"
        clock_face_color = "black"
        hour_tick_color = "black"
        sec_hand_color = "red"
        min_hand_color = "black"
        hour_hand_color = "black"

        # Hand lengths and widths (relative to radius)
        sec_hand_len = radius * 0.85
        min_hand_len = radius * 0.75
        hour_hand_len = radius * 0.55
        sec_hand_width = 1
        min_hand_width = 2
        hour_hand_width = 3

        # Create image
        img = Image.new('RGB', (img_size, img_size), color=bg_color)
        draw = ImageDraw.Draw(img)

        # Draw clock face circle
        draw.ellipse(
            (center[0] - radius, center[1] - radius,
             center[0] + radius, center[1] + radius),
            outline=clock_face_color, width=2
        )

        # Draw hour tick marks
        for i in range(1, 13): # Draw 12 ticks
            angle = math.radians((i / 12) * 360 - 90) # Start from 12 o'clock
            tick_len = 5 if i % 3 != 0 else 10 # Longer ticks for 3, 6, 9, 12
            tick_width = 1 if i % 3 != 0 else 2
            start_x = center[0] + (radius - tick_len) * math.cos(angle)
            start_y = center[1] + (radius - tick_len) * math.sin(angle)
            end_x = center[0] + radius * math.cos(angle)
            end_y = center[1] + radius * math.sin(angle)
            draw.line([(start_x, start_y), (end_x, end_y)], fill=hour_tick_color, width=tick_width)

        # Calculate hand angles
        # Hours: Use modulo 12, but 12 o'clock is 0 in angle calculations
        h = self.time_obj.hours % 12
        m = self.time_obj.minutes
        s = self.time_obj.seconds

        sec_angle = math.radians((s / 60) * 360 - 90)
        min_angle = math.radians(((m + s / 60) / 60) * 360 - 90)
        # Hour angle depends on hours, minutes, and seconds
        hour_fraction = (h + m / 60 + s / 3600)
        # Normalize hour fraction to the 0-12 range for angle calculation
        # (e.g., 12 maps to 0)
        hour_angle = math.radians(((hour_fraction % 12) / 12) * 360 - 90)


        # Calculate hand end points
        sec_end_x = center[0] + sec_hand_len * math.cos(sec_angle)
        sec_end_y = center[1] + sec_hand_len * math.sin(sec_angle)
        min_end_x = center[0] + min_hand_len * math.cos(min_angle)
        min_end_y = center[1] + min_hand_len * math.sin(min_angle)
        hour_end_x = center[0] + hour_hand_len * math.cos(hour_angle)
        hour_end_y = center[1] + hour_hand_len * math.sin(hour_angle)

        # Draw hands
        draw.line([center, (sec_end_x, sec_end_y)], fill=sec_hand_color, width=sec_hand_width)
        draw.line([center, (min_end_x, min_end_y)], fill=min_hand_color, width=min_hand_width)
        draw.line([center, (hour_end_x, hour_end_y)], fill=hour_hand_color, width=hour_hand_width)

        # Central pin
        pin_radius = hour_hand_width
        draw.ellipse(
            (center[0] - pin_radius, center[1] - pin_radius,
            center[0] + pin_radius, center[1] + pin_radius),
            fill=hour_hand_color
        )

        # Save the image
        img = img.resize((224, 224), Image.LANCZOS)
        img.save(filename)
        return img

if __name__ == "__main__":
    test_time_str = "[03:45:10]"
    parsed_time = TimeObj.from_string(test_time_str)
    if parsed_time:
        print(f"Parsed time: {parsed_time}")
        clock_gen = ClockGen(parsed_time)
        clock_gen.generate_clock("test_clock.png")

        random_time = TimeObj()
        print(f"Random time: {random_time}")
        clock_gen_random = ClockGen(random_time)
        clock_gen_random.generate_clock("random_clock.png")

        time1 = TimeObj(3, 0, 0)
        time2 = TimeObj(3, 0, 30)
        diff_sec, diff_dict = time1.subtract(time2)
        print(f"Difference between {time1} and {time2}: {diff_sec} seconds ({diff_dict['formatted']})")

        time3 = TimeObj(1, 0, 0)
        time4 = TimeObj(11, 0, 0)
        diff_sec, diff_dict = time3.subtract(time4)
        print(f"Difference between {time3} and {time4}: {diff_sec} seconds ({diff_dict['formatted']})") # Should be 2 hours

        time5 = TimeObj(12, 0, 0)
        time6 = TimeObj(6, 0, 0)
        diff_sec, diff_dict = time5.subtract(time6)
        print(f"Difference between {time5} and {time6}: {diff_sec} seconds ({diff_dict['formatted']})") # Should be 6 hours
    else:
        print(f"Failed to parse time string: {test_time_str}") 