import Paramter
from PSRmodel import CompressedPSR
if __name__ == "__main__":
    file = "Setting\\Shuttle.json"
    Paramter.readfile(file=file)
    c = CompressedPSR("Shuttle")
    c.LoadedExistedModel(file="ShuttleRewards.json")
    c.saveModel(epoch=0)