import jiwer

def detector(trainSet, liveSet, parameters):
    firstColumn = liveSet.axes[1][0]
    secondColumn = liveSet.axes[1][1]
    transformation = jiwer.Compose([
        jiwer.ToLowerCase(),
        jiwer.RemoveWhiteSpace(replace_by_space=True),
        jiwer.RemovePunctuation(),
        jiwer.ReduceToListOfListOfWords()
    ])

    wer = jiwer.wer(liveSet[firstColumn].tolist(), liveSet[secondColumn].tolist(), truth_transform=transformation, hypothesis_transform=transformation)
    threshold = float (parameters.get("threshold", 0.3))
    return int(wer>threshold), wer