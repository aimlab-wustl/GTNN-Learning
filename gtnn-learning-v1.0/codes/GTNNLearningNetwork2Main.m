% -------------------------------------------------------------------------
% GTNNLearning - A sparsity-driven learning framework for training Growth 
% Transform Neural Networks (GTNN)
% -------------------------------------------------------------------------
% Uses the following data for training and evaluation
% with the assumption that the training data is stored ../data/
% trainx -> input data (number of data points x feature dimension)
% Ytrain -> input labels (number of data points x number of classes)
% valx -> validation data (number of data points x feature dimension)
% Yval -> validation labels (number of data points x number of classes)
% testx -> test data (number of data points x feature dimension)
% Ytest -> test labels (number of data points x number of classes)
% Training, validation and test labels should be M-dimensional vectors
% where M is the number of classes.
% An example for a three class label is [-1 -1 1] to indicate
% that the training label belongs to class 3. 
% In case there is no validation data, programs sets aside a percentage of 
% data points for validation.
% -------------------------------------------------------------------------
% Copyright (C) [2021] Washington University in St. Louis
% Version: 1.0
% Created by: [Ahana Gangopadhyay and Shantanu Chakrabartty]
% -------------------------------------------------------------------------
% Citations for this tool are: 
% 1. Ahana Gangopadhyay and Shantanu Chakrabartty (2021). A Sparsity-driven 
% Backpropagation-less Learning Framework using Populations of Spiking 
% Growth Transform Neurons. Frontiers in Neuroscience, 
% doi: 10.3389/fnins.2021.715451.
% 2. Ahana Gangopadhyay, Darshit Mehta and Shantanu Chakrabartty. (2020). 
% A Spiking Growth Transform Neuron and Population Model. 
% Frontiers in Neuroscience, Vol. 14, page 425.
% -------------------------------------------------------------------------
% Washington University hereby grants to you a non-transferable, 
% non-exclusive, royalty-free, non-commercial, research license to use and 
% copy the computer code provided here (the “Software”).  You agree to 
% include this license and the above copyright notice in all copies of the 
% Software.  The Software may not be distributed, shared, or transferred to 
% any third party.  This license does not grant any rights or licenses to 
% any other patents, copyrights, or other forms of intellectual property 
% owned or controlled by Washington University.  If interested in obtaining 
% a commercial license, please contact Washington University's Office of 
% Technology Management (otm@dom.wustl.edu).
% -------------------------------------------------------------------------
% YOU AGREE THAT THE SOFTWARE PROVIDED HEREUNDER IS EXPERIMENTAL AND IS 
% PROVIDED “AS IS”, WITHOUT ANY WARRANTY OF ANY KIND, EXPRESSED OR IMPLIED, 
% INCLUDING WITHOUT LIMITATION WARRANTIES OF MERCHANTABILITY OR FITNESS FOR 
% ANY PARTICULAR PURPOSE, OR NON-INFRINGEMENT OF ANY THIRD-PARTY PATENT, 
% COPYRIGHT, OR ANY OTHER THIRD-PARTY RIGHT.  IN NO EVENT SHALL THE CREATORS 
% OF THE SOFTWARE OR WASHINGTON UNIVERSITY BE LIABLE FOR ANY DIRECT, 
% INDIRECT, SPECIAL, OR CONSEQUENTIAL DAMAGES ARISING OUT OF OR IN ANY WAY 
% CONNECTED WITH THE SOFTWARE, THE USE OF THE SOFTWARE, OR THIS AGREEMENT, 
% WHETHER IN BREACH OF CONTRACT, TORT OR OTHERWISE, EVEN IF SUCH PARTY IS 
% ADVISED OF THE POSSIBILITY OF SUCH DAMAGES. YOU ALSO AGREE THAT THIS 
% SOFTWARE WILL NOT BE USED FOR CLINICAL PURPOSES.
% -------------------------------------------------------------------------


%% Load the training data - Make sure the file exists
load '../data/ucsd-gsd/10shots/batch4';
% load '../data/linear2class2D';
% load '../data/cleanXOR2D';
% load '../data/noisyXOR2D';


%% Design network and set training hyper-parameters
%% You can also load hyper-parameters used to reproduce results on UCSD GSD dataset by uncommenting line 80
% network.num_layers = 2;               % Number of layers
% network.num_sub = [20, 1];            % Number of sub-networks in each layer
% network.density = [1, 1];             % Specify level of sparsity in each layer (1 --> fully-connected); vector dimension should be equal to network.num_layers
% network.network_type = [1, 1];        % Specify network type for each layer; 1: fully-connected, 2: feed-forward; vector dimension should be equal to network.num_layers
% network.last_layer = 1;               % 1 --> Considers only spikes from last layer for inference, useful when Y has no linear relationship with X
% network.include_labels = [0, 1];      % 1 --> Include labels in layer-wise training
% 
% hyperparams.improv_epochs = 6;        % Early stopping criterion based on performance on validation set
% hyperparams.maxiter = 200;            % Number of training iterations per data point per epoch
% hyperparams.eta = [0, 0.0005];        % Learning rate in each layer; vector dimension should be equal to network.num_layers
% hyperparams.trainEpochs = 1;          % Trains layers 1 to N-1 for these many epochs; set to a large number if they are to be trained throughout
% hyperparams.shuffleFlag = 1;          % Shuffle training data

load ucsd-gsd-nw2-hyperparams

%% Preprocessing
if exist('valx', 'var')==0
    valx = []; Yval = [];
end
shuffleFlag = 0;                       % Shuffle training data, keep 0 to reproduce results on UCSD GSD dataset
valData = 0.5;                         % Percentage of training data set aside for validation
scaleFlag = 1;                         % Scale training data between specified range
range = [0, 1];
procData = GTNNLearningDataPreprocess(trainx, Ytrain, valx, Yval, testx, Ytest, hyperparams.shuffleFlag, valData, scaleFlag, range);


%% Plotting results and contours
flag.plotFlag = 1;              % Plot accuracy and sparsity metric versus epochs
flag.contourFlag = 1;           % Plot classification boundary (only for 2D data)


%% Train network
fprintf('Start GTNN training ...');
[network, trainedNetwork, trainResults] = GTNNLearningTrain(network, hyperparams, procData, flag);
fprintf('...done\n');


%% Evaluate performance on test set
testResults = GTNNLearningTest(network, hyperparams, trainedNetwork, procData);

%% Contour plot
if flag.contourFlag == 1 && size(procData.trainx, 2) == 2
   fprintf('\n Plotting Contour ....');
   GTNNLearningContour(network, hyperparams, trainedNetwork, procData, range, 0.01);
   fprintf('....done\n');
end