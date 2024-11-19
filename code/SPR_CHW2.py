from sklearn.datasets import fetch_openml
from numpy import dot, linalg, mean
from pandas import DataFrame, concat
import matplotlib.pyplot as plt
from sklearn.model_selection import LeaveOneOut
from sklearn.neighbors import KNeighborsClassifier
import time

dataSize = 10000
kfold = 10


mnist = fetch_openml('mnist_784')
mnistData: DataFrame = mnist.data[:dataSize]
mnistTarget: DataFrame = mnist.target[:dataSize]

trainSet = mnistData[:dataSize*8//10]
testSet = mnistData[dataSize*8//10:]


def metric(q, p):
    dotProduct = dot(q, p)
    magnitudeProduct = linalg.norm(q) * linalg.norm(p)
    return dotProduct / magnitudeProduct


def predict(x: DataFrame, trainSet: DataFrame, k: int):
    kNearestNeighbors = []

    for i in range(len(trainSet.index)):
        targetIndex = trainSet.index[i]
        sim = metric(x.to_numpy(), trainSet.iloc[i].to_numpy())
        kNearestNeighbors = sorted(
            kNearestNeighbors, key=lambda neighbor: -neighbor["sim"])

        if len(kNearestNeighbors) < k:
            kNearestNeighbors.append(
                {"label": mnistTarget.loc[targetIndex], "sim": sim})
        elif sim > kNearestNeighbors[-1]["sim"]:
            kNearestNeighbors.pop()
            kNearestNeighbors.append(
                {"label": mnistTarget.loc[targetIndex], "sim": sim})

    counts = [0] * 10
    for neighbor in kNearestNeighbors:
        counts[int(neighbor["label"])] += 1
    maxCount = max(counts)
    label = [i for i, count in enumerate(counts) if count == maxCount]

    return str(label[0])


def Kaccuracy(k: int):
    accuracy = []
    for i in range(kfold):
        firstIndex = i*(len(trainSet.index)//kfold)
        lastIndexIndex = (i+1)*(len(trainSet.index)//kfold)

        kfoldDataSet = trainSet[firstIndex: lastIndexIndex]
        newTrainSet = concat(
            [trainSet[0: firstIndex], trainSet[lastIndexIndex:]])

        rightClassification = 0
        for i in range(len(kfoldDataSet.index)):
            predictLabel = predict(kfoldDataSet.iloc[i], newTrainSet, k)

            realLabel = mnistTarget[kfoldDataSet.index[i]]
            if (predictLabel == realLabel):
                rightClassification += 1

        accuracy.append(
            float("%.2f" % (rightClassification / len(kfoldDataSet.index))))

    return float("%.2f" % mean(accuracy))


accuracies = []
usedK = []
for i in range(15):
    accuracies.append(Kaccuracy((2*i)+1))
    usedK.append((2*i)+1)

plt.figure(figsize=(10, 5))
plt.plot(usedK, accuracies, marker='o', linestyle='dashed', color='b')
plt.title('Cross validation scores for different k')
plt.xlabel('K')
plt.ylabel('Cross validation accuracy')
plt.grid(True)
plt.show()

accuracies = []
usedK = []

for i in range(15):
    k = (i*2) + 1

    usedK.append(k)

    knn = KNeighborsClassifier(n_neighbors=k, metric="cosine")
    knn.fit(trainSet, mnistTarget.loc[trainSet.index])
    predicts = knn.predict(testSet)

    rightClassification = 0
    for j in range(len(testSet.index)):
        predictLabel = predicts[j]
        realLabel = mnistTarget[testSet.index[j]]
        if (predictLabel == realLabel):
            rightClassification += 1

    accuracies.append(
        (rightClassification / len(testSet.index)))


def removeAmbiguousSamples(dataSet, target, k):
    knn = KNeighborsClassifier(n_neighbors=k, metric="cosine")
    loo = LeaveOneOut()
    ambiguousIndices = []

    for trainIndex, testIndex in loo.split(dataSet):
        trainData, testSample = dataSet.iloc[trainIndex], dataSet.iloc[testIndex]
        trainTarget, testTarget = target.iloc[trainIndex], target.iloc[testIndex]

        knn.fit(trainData, trainTarget)
        predict = knn.predict(testSample)

        if predict[0] != testTarget.values[0]:
            ambiguousIndices.append(testIndex[0])

    ambiguousIndices = list(set(ambiguousIndices))

    cleanedDataSet = dataSet.drop(index=ambiguousIndices)
    cleanedTarget = target.drop(index=ambiguousIndices)

    return cleanedDataSet, cleanedTarget


cleanedDataSet, cleanedTarget = removeAmbiguousSamples(
    trainSet, mnistTarget, 1)


start = time.time()

k = 1

knn = KNeighborsClassifier(n_neighbors=k, metric="cosine")
knn.fit(trainSet, mnistTarget.loc[trainSet.index])
predicts = knn.predict(testSet)

rightClassification = 0
for j in range(len(testSet.index)):
    predictLabel = predicts[j]
    realLabel = mnistTarget[testSet.index[j]]
    if (predictLabel == realLabel):
        rightClassification += 1

print(
    (rightClassification / len(testSet.index)))

end = time.time()
print("time measure in cleaned dataset", end - start)


start = time.time()

k = 1
knn = KNeighborsClassifier(n_neighbors=k, metric="cosine")
knn.fit(cleanedDataSet, cleanedTarget.loc[cleanedDataSet.index])
predicts = knn.predict(testSet)

rightClassification = 0
for j in range(len(testSet.index)):
    predictLabel = predicts[j]
    realLabel = mnistTarget[testSet.index[j]]
    if (predictLabel == realLabel):
        rightClassification += 1

print(
    (rightClassification / len(testSet.index)))

end = time.time()
print("time measure in cleaned dataset", end - start)
