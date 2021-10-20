package main

import "github.com/LdDl/cnns"

var (
	imgWidth    = 448
	imgHeight   = 448
	numOfEpochs = 50
	modelPath   = "model/cnnModel-50e"
)

func main() {
	//model := importModelFormFile()
	//predict(&model, "C:\\Users\\onefo\\Project\\image-learning\\Brain-Tumor-Classification-DataSet\\Training\\no_tumor\\image(1).jpg")
	model := cnnTrain(imgWidth, imgHeight, numOfEpochs)
	err := model.ExportToFile(modelPath, true)
	if err != nil {
		panic(err)
	}

}

func importModelFormFile() cnns.WholeNet {
	model := cnns.WholeNet{
		LP: cnns.NewLearningParametersDefault(),
	}
	err := model.ImportFromFile(modelPath, true)
	if err != nil {
		panic(err)
	}
	return model
}
