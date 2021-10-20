package main

import (
	"fmt"
	"github.com/nfnt/resize"
	"gocv.io/x/gocv"
	"gonum.org/v1/gonum/mat"
	"image/jpeg"
	"io/ioutil"
	"log"
	"os"
)

var directoryPath = "C:\\Users\\onefo\\Project\\image-learning\\Brain-Tumor-Classification-DataSet\\%s"

func Preparation() ([]*mat.Dense, []*mat.Dense, []*mat.Dense, []*mat.Dense) {
	var (
		trainX []*mat.Dense
		trainY []*mat.Dense
		testX  []*mat.Dense
		testY  []*mat.Dense
	)
	loadGliomaTumor(&trainX, &trainY, &testX, &testY)
	loadMeningiomaTumor(&trainX, &trainY, &testX, &testY)
	loadNoTumor(&trainX, &trainY, &testX, &testY)
	loadPituitaryTumor(&trainX, &trainY, &testX, &testY)
	return trainX, trainY, testX, testY
}

func loadGliomaTumor(trainX *[]*mat.Dense, trainY *[]*mat.Dense, testX *[]*mat.Dense, testY *[]*mat.Dense) {
	trainPath := "Training\\glioma_tumor"
	testPath := "Testing\\glioma_tumor"
	loadTools(fmt.Sprintf(directoryPath, trainPath), trainX, trainY, []float64{1.0, 0.0, 0.0, 0.0})
	loadTools(fmt.Sprintf(directoryPath, testPath), testX, testY, []float64{1.0, 0.0, 0.0, 0.0})
}
func loadMeningiomaTumor(trainX *[]*mat.Dense, trainY *[]*mat.Dense, testX *[]*mat.Dense, testY *[]*mat.Dense) {
	trainPath := "Training\\meningioma_tumor"
	testPath := "Testing\\meningioma_tumor"
	loadTools(fmt.Sprintf(directoryPath, trainPath), trainX, trainY, []float64{0.0, 1.0, 0.0, 0.0})
	loadTools(fmt.Sprintf(directoryPath, testPath), testX, testY, []float64{0.0, 1.0, 0.0, 0.0})
}
func loadNoTumor(trainX *[]*mat.Dense, trainY *[]*mat.Dense, testX *[]*mat.Dense, testY *[]*mat.Dense) {
	trainPath := "Training\\no_tumor"
	testPath := "Testing\\no_tumor"
	loadTools(fmt.Sprintf(directoryPath, trainPath), trainX, trainY, []float64{0.0, 0.0, 1.0, 0.0})
	loadTools(fmt.Sprintf(directoryPath, testPath), testX, testY, []float64{0.0, 0.0, 1.0, 0.0})
}
func loadPituitaryTumor(trainX *[]*mat.Dense, trainY *[]*mat.Dense, testX *[]*mat.Dense, testY *[]*mat.Dense) {
	trainPath := "Training\\pituitary_tumor"
	testPath := "Testing\\pituitary_tumor"
	loadTools(fmt.Sprintf(directoryPath, trainPath), trainX, trainY, []float64{0.0, 0.0, 0.0, 1.0})
	loadTools(fmt.Sprintf(directoryPath, testPath), testX, testY, []float64{0.0, 0.0, 0.0, 1.0})
}

func loadTools(dirPath string, data *[]*mat.Dense, target *[]*mat.Dense, targetFloat []float64) {
	rd, _ := ioutil.ReadDir(dirPath)
	for _, fi := range rd {
		if fi.IsDir() {
			continue
		} else {
			fullPath := fmt.Sprintf(dirPath+"\\%s", fi.Name())
			//reSize(fullPath)
			*data = append(*data, img2Dense(fullPath))
			*target = append(*target, mat.NewDense(4, 1, targetFloat))
		}
	}
}

func img2Dense(imgPath string) *mat.Dense {
	img := gocv.IMRead(imgPath, gocv.IMReadAnyColor)
	convertedImg := gocv.NewMat()
	img.ConvertTo(&convertedImg, gocv.MatTypeCV64F)
	imgArray, err := convertedImg.DataPtrFloat64()
	if err != nil {
		panic(err)
	}
	dense := mat.NewDense(imgWidth, imgHeight, imgArray)
	return dense
}

func reSize(imgPath string) {
	file, err := os.Open(imgPath)
	if err != nil {
		panic(err)
	}
	img, err := jpeg.Decode(file)
	if err != nil {
		log.Fatal(err)
	}
	file.Close()
	m := resize.Resize(448, 448, img, resize.Lanczos3)
	err = os.Remove(imgPath)
	out, err := os.Create(imgPath)
	defer out.Close()
	if err = jpeg.Encode(out, m, nil); err != nil {
		panic(err)
	}
}
