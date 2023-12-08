from dataclasses import dataclass


@dataclass(frozen=True, order=True)
class DataBlock:
    digit: int  # 0, 1, 2, ..., 9
    index: int  # 0, 1, 2, ..., 9
    gender: str  # M or F
    mfccs: list  # [[mfccs], [mfccs], [mfccs], ...]

    def filter_mfccs(self, mfcc_indexes):
        return DataBlock(self.digit, self.index, self.gender, [[mfcc[i] for i in mfcc_indexes if i >= 0] for mfcc in self.mfccs])

    def __str__(self):
        return str(self.digit) + " " + str(self.gender) + " " + str(self.index) + " MFCCs..."
