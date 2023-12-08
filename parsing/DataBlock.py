from dataclasses import dataclass


@dataclass(frozen=True, order=True)
class DataBlock:
    digit: int  # 0, 1, 2, ..., 9
    index: int  # 0, 1, 2, ..., 9
    gender: str  # m or f
    mfccs: list  # [[mfccs], [mfccs], [mfccs], ...]

    def filter_mfccs(self, mfcc_indexes):
        return DataBlock(self.digit, self.index, self.gender, [[mfcc[i] for i in mfcc_indexes] for mfcc in self.mfccs])

    def __str__(self):
        return str(self.digit) + " " + str(self.gender) + " " + str(self.index) + " MFCCs..."
