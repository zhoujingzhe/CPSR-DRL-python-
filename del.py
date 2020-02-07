import os

if __name__ == "__main__":
    mainPath = "folder"
    Epoches = os.listdir(mainPath)
    for epoch in Epoches:
        files = os.listdir(mainPath + "\\" + epoch)
        for file in files:
            if file != "mInf.txt" and file != "pv.txt" and file != "aoMats.txt":
                os.remove(mainPath + "\\" + epoch + "\\" + file)