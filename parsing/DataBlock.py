from dataclasses import dataclass


@dataclass(frozen=True, order=True)
class DataBlock:
    digit: int  # 0, 1, 2, ..., 9
    gender: str  # m or f
    index: int  # 0, 1, 2, ..., 9
    mfccs: list  # [[mfccs], [mfccs], [mfccs], ...]

    def __str__(self):
        return str(self.digit) + " " + str(self.gender) + " " + str(self.index) + " MFCCs..."
