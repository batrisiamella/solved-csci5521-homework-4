Download Link: https://assignmentchef.com/product/solved-csci5521-homework-4
<br>
<ol>

 <li>Let X = {<strong>x</strong><sup>1</sup><em>,…,</em><strong>x</strong><em><sup>N</sup></em>} with <strong>x</strong><em><sup>t </sup></em>∈ R<em><sup>D</sup>,t </em>= 1<em>,…,N </em>be a given training set. Assume that the dataset is centered, i.e., . We focus on performing linear dimensionality reduction on the dataset using PCA (principal component analysis). With PCA, for each <strong>x</strong><em><sup>t </sup></em>∈ R<em><sup>D</sup></em>, we get <strong>z</strong><em><sup>t </sup></em>= <em>W</em><strong>x</strong><em><sup>t</sup></em>, where <strong>z</strong><em><sup>t </sup></em>∈ R<em><sup>d</sup>,d &lt; D</em>, is the low dimensional projection, and <em>W </em>∈ R<em><sup>d</sup></em><sup>×<em>D </em></sup>is the PCA projection matrix. Let Σ = be the sample covariance matrix. Further, let <strong>v</strong><em><sup>t </sup></em>= <em>W<sup>T </sup></em><strong>z</strong><em><sup>t </sup></em>so that <strong>v</strong><em><sup>t </sup></em>∈ R<em><sup>D</sup></em>.

  <ul>

   <li>Professor HighLowHigh claims: <strong>v</strong><em><sup>t </sup></em>= <strong>x</strong><em><sup>t </sup></em>for all <em>t </em>= 1<em>,…,N</em>. Is the claim correct? Clearly explain and prove your answer with necessary (mathematical) details.</li>

   <li>Professor HighLowHigh also claims:</li>

  </ul></li>

</ol>

<em>N                             N                             N</em>

Xk<strong>x</strong><em>t</em>k22 − Xk<strong>v</strong><em>t</em>k22 = Xk<strong>x</strong><em>t </em>− <strong>v</strong><em>t</em>k22 <em>,</em>

<em>t</em>=1                         <em>t</em>=1                          <em>t</em>=1

where for a vector <strong>a </strong>. Is the claim correct? Clearly explain and prove your answer with necessary (mathematical) details.

<ol start="2">

 <li>Let Z = {(<strong>x</strong><sup>1</sup><em>,</em><strong>r</strong><sup>1</sup>)<em>,…,</em>(<strong>x</strong><em><sup>N</sup>,</em><strong>r</strong><em><sup>N</sup></em>)}<em>,</em><strong>x</strong><em><sup>t </sup></em>∈ R<em><sup>d</sup>,</em><strong>r</strong><em><sup>t </sup></em>∈ R<em><sup>k </sup></em>be a set of <em>N </em>training samples. We consider training a multilayer perceptron as shown in Figure 1. We consider a general setting where the transfer functions at each stage are denoted by <em>g</em>, i.e.,</li>

</ol>

and  <em> ,</em>

where <em>a<sup>t</sup><sub>h</sub>,a<sup>t</sup><sub>i </sub></em>respectively denote the input activation for hidden node <em>h </em>and output node <em>i</em>. Further, let <em>L</em>(·<em>,</em>·) be the loss function, so that the learning focuses on minimizing:

<em>N       k</em>

<em>E</em>(<em>W,V </em>|Z) = <sup>XX</sup><em>L</em>(<em>r<sub>i</sub><sup>t</sup>,y<sub>i</sub><sup>t</sup></em>) <em>.</em>

<em>t</em>=1 <em>i</em>=1

<ul>

 <li>Show that the stochastic gradient descent update for <em>v<sub>i,h </sub></em>is of the form <em>v</em><em>i,h</em>new = <em>v</em><em>i,h</em>old + ∆<em>v</em><em>i,h</em>, with the update</li>

</ul>

∆<em>v<sub>i,h </sub></em>= <em>η</em>∆<em><sup>t</sup><sub>i</sub>z<sub>h</sub><sup>t </sup>,          </em>where ∆<em> .        </em>(1)

<ul>

 <li>Show that the stochastic gradient descent update for <em>w<sub>h,j </sub></em>is of the form <em>w</em><em>h,j</em>new = <em>w</em><em>h,j</em>old + ∆<em>w</em><em>h,j</em>, with the update</li>

</ul>

∆<em>w<sub>h,j </sub></em>= <em>η</em>∆<em><sup>t</sup><sub>h</sub>x<sup>t</sup><sub>j </sub>,           </em>where ∆<em> .       </em>(2)

Figure 1: Two layer perceptron.

<strong>Programming assignment:</strong>

The next problem involves programming. For Question 3, we will be using the 2-class classification datasets from Boston50 and Boston75. In particular, we will develop code for 2-class Support Vector Machines (SVMs) using gradient descent. The goal will be to modify your code for MyLogisticReg2 from HW3.

<ol start="3">

 <li>We will develop code for 2-class SVMs with parameters (<strong>w</strong><em>,w</em><sub>0</sub>) where <strong>w </strong>∈ R<em><sup>d</sup>,w</em><sub>0 </sub>∈ R. Assume a given dataset {(<strong>x</strong><em><sup>t</sup>,y<sup>t</sup></em>)<em>,t </em>= 1<em>,…,N</em>}, where <strong>x</strong><em><sup>t </sup></em>∈ R<em><sup>d </sup></em>and <em>y<sup>t </sup></em>∈ {−1<em>,</em>1}. Recall from our discussion in class that training SVMs involves minimizing the following objective function:</li>

</ol>

<em> .         </em>(3)

We will use <em>λ </em>= 5 in this assignment.

For reference, compare the objective function to that of regularized logistic regression which you recently worked on as part of HW3:

<em> ,                                                                                                    </em>(4)

where we had used <em>λ </em>= 0 for the HW3 code.

We will develop code for MySVM2 with corresponding MySVM2.fit(X,y) and MySVM2.predict(X) functions. Parameters for the model can be initialized following what you had done for MyLogisticReg2. In the fit function, the parameters will be estimated using <em>mini-batch stochastic gradient descent </em>with different mini-batch sizes <em>m </em>≤ <em>n</em>. In particular, you will modify your MyLogisticReg2 code by using gradients for the SVM objective in (3) instead of the logistic regression objective in (4). Further, you will have to add the mini-batch stochastic gradient descent (SGD) functionality which, for a pre-specified mini-batch size <em>m</em>, picks <em>m </em>unique points at random to do the gradient descent in each iteration. We will run experiments with different values of <em>m</em>.

We will compare the performance of MySVM2 for different values of mini-batch size <em>m </em>with

LogisticRegression<a href="#_ftn1" name="_ftnref1"><sup>[1]</sup></a> on two datasets: Boston50 and Boston75. Recall that Boston has 506 data points, and a 5-fold cross-validation leaves <em>n </em>≈ 400 points for training in each fold.<a href="#_ftn2" name="_ftnref2"><sup>[2]</sup></a>For mini-batch SGD, we will consider three different values of <em>m</em>:

(i) <em>m </em>= 40, which is ≈ 10% of the dataset in each fold for 5-fold cross-validation, (ii) <em>m </em>= 200, which is ≈ 50% of the dataset in each fold for 5-fold cross-validation, and (iii) <em>m </em>= <em>n</em>, which is the full dataset in each fold for 5-fold cross-validation.

Note that <em>m </em>= <em>n </em>uses the full dataset (available for that fold) in each iteration and hence corresponds to the usual gradient descent.<a href="#_ftn3" name="_ftnref3"><sup>[3]</sup></a>

Using my cross val with 5-fold cross-validation, report the error rates in each fold as well as the mean and standard deviation of error rates across all folds for the four methods: MySVM2 with <em>m </em>= 40<em>,m </em>= 200, and <em>m </em>= <em>n</em>, and LogisticRegression, applied to the two 2-class classification datasets: Boston50 and Boston75.

You will have to submit (a) <strong>code </strong>and (b) <strong>summary of results</strong>:

<ul>

 <li><strong>Code</strong>: You will have to submit code for MySVM2() as well as a wrapper code q3().</li>

</ul>

For developing MySVM2(), you are encouraged to consult the code for MyLogisticReg2() from HW3. You need to make sure you have init , fit, and predict implemented in MySVM2. init (d,m) will initialize the parameters and will take the data dimensionality <em>d </em>and mini-batch size <em>m </em>as input. You can add additional inputs such as the step size or the convergence threshold. fit(X,y) will take the data features <em>X </em>and labels <em>y </em>and will use mini-batch SGD to estimate the parameters <strong>w</strong><em>,w</em><sub>0</sub>. predict(X) will take a feature matrix corresponding to the test set and return the predicted labels. Your class MySVM2() will <strong>NOT </strong>inherit any base class in sklearn.

<strong>The wrapper code </strong>(main file) has no input and is used to prepare the datasets, and make calls to my cross val(method,<em>X</em>,<strong>y</strong>,<em>k</em>) to generate the error rate results for each dataset and each method. The code for my cross val(method,<em>X</em>,<strong>y</strong>,<em>k</em>) must be yours (e.g., code you made in HW1 with modifications as needed) and you cannot use cross val score() in sklearn. The results should be printed to terminal (not generating an additional file in the folder). Make sure the calls to my cross val(method,<em>X</em>,<strong>y</strong>,<em>k</em>) are made in the following order and add a print to the terminal before each call to show which method and dataset is being used:

<ol>

 <li>MySVM2 with <em>m </em>= 40 for Boston50;</li>

 <li>MySVM2 with <em>m </em>= 200 for Boston50;</li>

</ol>

<ul>

 <li>MySVM2 with <em>m </em>= <em>n </em>for Boston50; iv. LogisticRegression for Boston50;</li>

</ul>

<ol>

 <li>MySVM2 with <em>m </em>= 40 for Boston75;</li>

 <li>MySVM2 with <em>m </em>= 200 for Boston75;</li>

</ol>

<ul>

 <li>MySVM2 with <em>m </em>= <em>n </em>for Boston75;</li>

 <li>LogisticRegression for Boston75.</li>

</ul>

*For the wrapper code, you need to make a q3.py file for it, and one should be able to run your code by calling “python q3.py” in command line window.

<ul>

 <li><strong>Summary of results</strong>: For each dataset and each method, report the test set error rates for each of the <em>k </em>= 5 folds, the mean error rate over the <em>k </em>folds, and the standard deviation of the error rates over the <em>k </em> Make a table to present the results for each method and each dataset (4 tables in total). Each column of the table represents a fold and add two columns at the end to show the overall mean error rate and standard deviation over the <em>k </em>folds. For example:</li>

</ul>

<table width="365">

 <tbody>

  <tr>

   <td colspan="7" width="365">Error rates for MySVM2 with <em>m </em>= 40 for Boston50</td>

  </tr>

  <tr>

   <td width="56">Fold 1</td>

   <td width="56">Fold 2</td>

   <td width="56">Fold 3</td>

   <td width="56">Fold 4</td>

   <td width="56">Fold 5</td>

   <td width="51">Mean</td>

   <td width="35">SD</td>

  </tr>

  <tr>

   <td width="56">#</td>

   <td width="56">#</td>

   <td width="56">#</td>

   <td width="56">#</td>

   <td width="56">#</td>

   <td width="51">#</td>

   <td width="35">#</td>

  </tr>

 </tbody>

</table>

<a href="#_ftnref1" name="_ftn1">[1]</a> You should use LogisticRegression from sklearn, as we did for HW3. Note that Linear SVMs are implemented in sklearn as LinearSVC, but we will not use it since we have not discussed it in class. We will stick to

LogisticRegression for comparisons.

<a href="#_ftnref2" name="_ftn2">[2]</a> Note that we are denoting the number of training points available for training in each fold as <em>n</em>, which is smaller than the size of the full dataset.

<a href="#_ftnref3" name="_ftn3">[3]</a> The exact value of <em>n </em>may differ mildly across the 5-folds since 506 cannot be exactly divided by 5. Your code for HW3 is already doing these splits, so this aspect should not need additional effort. In the code, the <em>m </em>= <em>n </em>needs to be passed as a special option (say, <em>m </em>= 10<sup>6 </sup>or <em>m </em>=“all”) so the code knows it has use the the full dataset for that fold.