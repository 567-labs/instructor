from pydantic import BaseModel, Field, model_validator
from typing import Literal


# Turns out this doesn't work well. since longer videos will be HH:MM:SS
# but shorter videos will be MM:SS, and the language model does not do 00:MM:SS well
# then we run into issues where 2:00 is parsed as 200 seconds
class Segment(BaseModel):
    title: str = Field(..., description="The title of the segment")
    timestamp: str = Field(..., description="The timestamp of the event as HH:MM:SS")


# We fix this by doing twi things
# Tell the LMM which format it wants to use
# And then we use a custom parser to parse the timestamp
class SegmentWithTimestamp(BaseModel):
    title: str = Field(..., description="The title of the segment")
    time_format: Literal["HH:MM:SS", "MM:SS"] = Field(
        ..., description="The format of the timestamp"
    )
    timestamp: str = Field(
        ..., description="The timestamp of the event as either HH:MM:SS or MM:SS"
    )

    @model_validator(mode="after")
    def parse_timestamp(self):
        if self.time_format == "HH:MM:SS":
            hours, minutes, seconds = map(int, self.timestamp.split(":"))
        elif self.time_format == "MM:SS":
            hours, minutes, seconds = 0, *map(int, self.timestamp.split(":"))
        else:
            raise ValueError("Invalid time format, must be HH:MM:SS or MM:SS")

        # Normalize seconds and minutes
        total_seconds = hours * 3600 + minutes * 60 + seconds
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)

        if hours > 0:
            self.timestamp = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        else:
            self.timestamp = f"00:{minutes:02d}:{seconds:02d}"

        return self


if __name__ == "__main__":
    # Make tests
    # Test cases for SegmentWithTimestamp
    test_cases = [
        (
            SegmentWithTimestamp(
                title="Introduction", time_format="MM:SS", timestamp="00:30"
            ),
            "00:00:30",
        ),
        (
            SegmentWithTimestamp(
                title="Main Topic", time_format="HH:MM:SS", timestamp="00:15:45"
            ),
            "00:15:45",
        ),
        (
            SegmentWithTimestamp(
                title="Conclusion", time_format="MM:SS", timestamp="65:00"
            ),
            "01:05:00",
        ),
    ]

    for input_data, expected_output in test_cases:
        try:
            assert input_data.timestamp == expected_output
            print(f"Test passed: {input_data.timestamp} == {expected_output}")
        except AssertionError:
            print(f"Test failed: {input_data.timestamp} != {expected_output}")

    # > Test passed: 00:00:30 == 00:00:30
    # > Test passed: 00:15:45 == 00:15:45
    # > Test passed: 01:05:00 == 01:05:00
