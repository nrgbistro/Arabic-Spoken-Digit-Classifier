from dataclasses import dataclass


@dataclass(frozen=True, order=True)
class DataBlock:
    digit: int
    gender: str
    index: int
    mfccs: list

    def __str__(self):
        return str(self.digit) + " " + str(self.gender) + " " + str(self.index) + " " + str(self.mfccs)
