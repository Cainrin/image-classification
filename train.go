package main

import (
	"fmt"
	"log"
	"math/rand"
	"time"

	"github.com/LdDl/cnns"
	"github.com/LdDl/cnns/tensor"
)

func cnnTrain(imgWidth int, imgHeight int, numOfEpochs int) cnns.WholeNet {
	rand.Seed(time.Now().UnixNano())
	conv := cnns.NewConvLayer(&tensor.TDsize{X: imgHeight, Y: imgWidth, Z: 1}, 1, 3, 1)
	relu := cnns.NewReLULayer(conv.GetOutputSize())
	maxPool := cnns.NewPoolingLayer(relu.GetOutputSize(), 2, 2, "max", "valid")
	fullyConnected := cnns.NewFullyConnectedLayer(maxPool.GetOutputSize(), 4)

	fullyConnected.SetActivationFunc(cnns.ActivationSygmoid)
	fullyConnected.SetActivationDerivativeFunc(cnns.ActivationSygmoidDerivative)

	net := cnns.WholeNet{
		LP: cnns.NewLearningParametersDefault(),
	}
	net.Layers = append(net.Layers, conv)
	net.Layers = append(net.Layers, relu)
	net.Layers = append(net.Layers, maxPool)
	net.Layers = append(net.Layers, fullyConnected)

	// Init train and test data
	trainX, trainY, testX, testY := Preparation()

	// Start traing process
	trainErr, testErr, err := net.Train(trainX, trainY, testX, testY, numOfEpochs)
	if err != nil {
		log.Printf("Can't train network due the error: %s", err.Error())
		panic(err)
	}
	fmt.Printf("Error on training data: %v\nError on test data: %v\n", trainErr, testErr)
	return net
}

func predict(net *cnns.WholeNet, imgPath string) {
	d := img2Dense(imgPath)
	err := net.FeedForward(d)
	if err != nil {
		log.Printf("Feedforward (testing) caused error: %s", err.Error())
		return
	}
	out := net.GetOutput()
	fmt.Println("\n>>>Out:")
	fmt.Println("\t", out.RawMatrix().Data)
}
