 ы&
ф§
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetypeѕ
Й
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ѕ
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshapeѕ"serve*2.2.02v2.2.0-rc4-8-g2b96f3662b8јЕ
ё
conv2d_10/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *!
shared_nameconv2d_10/kernel
}
$conv2d_10/kernel/Read/ReadVariableOpReadVariableOpconv2d_10/kernel*&
_output_shapes
: *
dtype0
t
conv2d_10/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_10/bias
m
"conv2d_10/bias/Read/ReadVariableOpReadVariableOpconv2d_10/bias*
_output_shapes
: *
dtype0
ј
batch_normalization_9/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_namebatch_normalization_9/gamma
Є
/batch_normalization_9/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_9/gamma*
_output_shapes
: *
dtype0
ї
batch_normalization_9/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *+
shared_namebatch_normalization_9/beta
Ё
.batch_normalization_9/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_9/beta*
_output_shapes
: *
dtype0
џ
!batch_normalization_9/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!batch_normalization_9/moving_mean
Њ
5batch_normalization_9/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_9/moving_mean*
_output_shapes
: *
dtype0
б
%batch_normalization_9/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *6
shared_name'%batch_normalization_9/moving_variance
Џ
9batch_normalization_9/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_9/moving_variance*
_output_shapes
: *
dtype0
ё
conv2d_11/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *!
shared_nameconv2d_11/kernel
}
$conv2d_11/kernel/Read/ReadVariableOpReadVariableOpconv2d_11/kernel*&
_output_shapes
:  *
dtype0
t
conv2d_11/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_11/bias
m
"conv2d_11/bias/Read/ReadVariableOpReadVariableOpconv2d_11/bias*
_output_shapes
: *
dtype0
љ
batch_normalization_10/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *-
shared_namebatch_normalization_10/gamma
Ѕ
0batch_normalization_10/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_10/gamma*
_output_shapes
: *
dtype0
ј
batch_normalization_10/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_namebatch_normalization_10/beta
Є
/batch_normalization_10/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_10/beta*
_output_shapes
: *
dtype0
ю
"batch_normalization_10/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"batch_normalization_10/moving_mean
Ћ
6batch_normalization_10/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_10/moving_mean*
_output_shapes
: *
dtype0
ц
&batch_normalization_10/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *7
shared_name(&batch_normalization_10/moving_variance
Ю
:batch_normalization_10/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_10/moving_variance*
_output_shapes
: *
dtype0
ё
conv2d_12/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *!
shared_nameconv2d_12/kernel
}
$conv2d_12/kernel/Read/ReadVariableOpReadVariableOpconv2d_12/kernel*&
_output_shapes
:  *
dtype0
t
conv2d_12/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_12/bias
m
"conv2d_12/bias/Read/ReadVariableOpReadVariableOpconv2d_12/bias*
_output_shapes
: *
dtype0
љ
batch_normalization_11/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *-
shared_namebatch_normalization_11/gamma
Ѕ
0batch_normalization_11/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_11/gamma*
_output_shapes
: *
dtype0
ј
batch_normalization_11/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_namebatch_normalization_11/beta
Є
/batch_normalization_11/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_11/beta*
_output_shapes
: *
dtype0
ю
"batch_normalization_11/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"batch_normalization_11/moving_mean
Ћ
6batch_normalization_11/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_11/moving_mean*
_output_shapes
: *
dtype0
ц
&batch_normalization_11/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *7
shared_name(&batch_normalization_11/moving_variance
Ю
:batch_normalization_11/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_11/moving_variance*
_output_shapes
: *
dtype0
ё
conv2d_13/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *!
shared_nameconv2d_13/kernel
}
$conv2d_13/kernel/Read/ReadVariableOpReadVariableOpconv2d_13/kernel*&
_output_shapes
:  *
dtype0
t
conv2d_13/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_13/bias
m
"conv2d_13/bias/Read/ReadVariableOpReadVariableOpconv2d_13/bias*
_output_shapes
: *
dtype0
љ
batch_normalization_12/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *-
shared_namebatch_normalization_12/gamma
Ѕ
0batch_normalization_12/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_12/gamma*
_output_shapes
: *
dtype0
ј
batch_normalization_12/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_namebatch_normalization_12/beta
Є
/batch_normalization_12/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_12/beta*
_output_shapes
: *
dtype0
ю
"batch_normalization_12/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"batch_normalization_12/moving_mean
Ћ
6batch_normalization_12/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_12/moving_mean*
_output_shapes
: *
dtype0
ц
&batch_normalization_12/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *7
shared_name(&batch_normalization_12/moving_variance
Ю
:batch_normalization_12/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_12/moving_variance*
_output_shapes
: *
dtype0
ё
conv2d_14/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *!
shared_nameconv2d_14/kernel
}
$conv2d_14/kernel/Read/ReadVariableOpReadVariableOpconv2d_14/kernel*&
_output_shapes
:  *
dtype0
t
conv2d_14/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_14/bias
m
"conv2d_14/bias/Read/ReadVariableOpReadVariableOpconv2d_14/bias*
_output_shapes
: *
dtype0
љ
batch_normalization_13/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *-
shared_namebatch_normalization_13/gamma
Ѕ
0batch_normalization_13/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_13/gamma*
_output_shapes
: *
dtype0
ј
batch_normalization_13/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_namebatch_normalization_13/beta
Є
/batch_normalization_13/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_13/beta*
_output_shapes
: *
dtype0
ю
"batch_normalization_13/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"batch_normalization_13/moving_mean
Ћ
6batch_normalization_13/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_13/moving_mean*
_output_shapes
: *
dtype0
ц
&batch_normalization_13/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *7
shared_name(&batch_normalization_13/moving_variance
Ю
:batch_normalization_13/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_13/moving_variance*
_output_shapes
: *
dtype0
ё
conv2d_15/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *!
shared_nameconv2d_15/kernel
}
$conv2d_15/kernel/Read/ReadVariableOpReadVariableOpconv2d_15/kernel*&
_output_shapes
:  *
dtype0
t
conv2d_15/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_15/bias
m
"conv2d_15/bias/Read/ReadVariableOpReadVariableOpconv2d_15/bias*
_output_shapes
: *
dtype0
љ
batch_normalization_14/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *-
shared_namebatch_normalization_14/gamma
Ѕ
0batch_normalization_14/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_14/gamma*
_output_shapes
: *
dtype0
ј
batch_normalization_14/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_namebatch_normalization_14/beta
Є
/batch_normalization_14/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_14/beta*
_output_shapes
: *
dtype0
ю
"batch_normalization_14/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"batch_normalization_14/moving_mean
Ћ
6batch_normalization_14/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_14/moving_mean*
_output_shapes
: *
dtype0
ц
&batch_normalization_14/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *7
shared_name(&batch_normalization_14/moving_variance
Ю
:batch_normalization_14/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_14/moving_variance*
_output_shapes
: *
dtype0
ё
conv2d_16/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *!
shared_nameconv2d_16/kernel
}
$conv2d_16/kernel/Read/ReadVariableOpReadVariableOpconv2d_16/kernel*&
_output_shapes
:  *
dtype0
t
conv2d_16/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_16/bias
m
"conv2d_16/bias/Read/ReadVariableOpReadVariableOpconv2d_16/bias*
_output_shapes
: *
dtype0
љ
batch_normalization_15/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *-
shared_namebatch_normalization_15/gamma
Ѕ
0batch_normalization_15/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_15/gamma*
_output_shapes
: *
dtype0
ј
batch_normalization_15/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_namebatch_normalization_15/beta
Є
/batch_normalization_15/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_15/beta*
_output_shapes
: *
dtype0
ю
"batch_normalization_15/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"batch_normalization_15/moving_mean
Ћ
6batch_normalization_15/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_15/moving_mean*
_output_shapes
: *
dtype0
ц
&batch_normalization_15/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *7
shared_name(&batch_normalization_15/moving_variance
Ю
:batch_normalization_15/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_15/moving_variance*
_output_shapes
: *
dtype0
ё
conv2d_17/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *!
shared_nameconv2d_17/kernel
}
$conv2d_17/kernel/Read/ReadVariableOpReadVariableOpconv2d_17/kernel*&
_output_shapes
: *
dtype0
t
conv2d_17/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_17/bias
m
"conv2d_17/bias/Read/ReadVariableOpReadVariableOpconv2d_17/bias*
_output_shapes
:*
dtype0
љ
batch_normalization_16/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namebatch_normalization_16/gamma
Ѕ
0batch_normalization_16/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_16/gamma*
_output_shapes
:*
dtype0
ј
batch_normalization_16/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_16/beta
Є
/batch_normalization_16/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_16/beta*
_output_shapes
:*
dtype0
ю
"batch_normalization_16/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"batch_normalization_16/moving_mean
Ћ
6batch_normalization_16/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_16/moving_mean*
_output_shapes
:*
dtype0
ц
&batch_normalization_16/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&batch_normalization_16/moving_variance
Ю
:batch_normalization_16/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_16/moving_variance*
_output_shapes
:*
dtype0

NoOpNoOp
Ої
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*Љї
valueєїBѓї BЩІ
╠
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer-6
layer-7
	layer_with_weights-4
	layer-8

layer_with_weights-5

layer-9
layer-10
layer-11
layer-12
layer-13
layer_with_weights-6
layer-14
layer_with_weights-7
layer-15
layer-16
layer_with_weights-8
layer-17
layer_with_weights-9
layer-18
layer-19
layer-20
layer_with_weights-10
layer-21
layer_with_weights-11
layer-22
layer-23
layer-24
layer-25
layer-26
layer_with_weights-12
layer-27
layer_with_weights-13
layer-28
layer-29
layer_with_weights-14
layer-30
 layer_with_weights-15
 layer-31
!layer-32
"regularization_losses
#trainable_variables
$	variables
%	keras_api
&
signatures
 
h

'kernel
(bias
)regularization_losses
*trainable_variables
+	variables
,	keras_api
Ќ
-axis
	.gamma
/beta
0moving_mean
1moving_variance
2regularization_losses
3trainable_variables
4	variables
5	keras_api
R
6regularization_losses
7trainable_variables
8	variables
9	keras_api
h

:kernel
;bias
<regularization_losses
=trainable_variables
>	variables
?	keras_api
Ќ
@axis
	Agamma
Bbeta
Cmoving_mean
Dmoving_variance
Eregularization_losses
Ftrainable_variables
G	variables
H	keras_api
R
Iregularization_losses
Jtrainable_variables
K	variables
L	keras_api
R
Mregularization_losses
Ntrainable_variables
O	variables
P	keras_api
h

Qkernel
Rbias
Sregularization_losses
Ttrainable_variables
U	variables
V	keras_api
Ќ
Waxis
	Xgamma
Ybeta
Zmoving_mean
[moving_variance
\regularization_losses
]trainable_variables
^	variables
_	keras_api
R
`regularization_losses
atrainable_variables
b	variables
c	keras_api
R
dregularization_losses
etrainable_variables
f	variables
g	keras_api
R
hregularization_losses
itrainable_variables
j	variables
k	keras_api
R
lregularization_losses
mtrainable_variables
n	variables
o	keras_api
h

pkernel
qbias
rregularization_losses
strainable_variables
t	variables
u	keras_api
Ќ
vaxis
	wgamma
xbeta
ymoving_mean
zmoving_variance
{regularization_losses
|trainable_variables
}	variables
~	keras_api
U
regularization_losses
ђtrainable_variables
Ђ	variables
ѓ	keras_api
n
Ѓkernel
	ёbias
Ёregularization_losses
єtrainable_variables
Є	variables
ѕ	keras_api
а
	Ѕaxis

іgamma
	Іbeta
їmoving_mean
Їmoving_variance
јregularization_losses
Јtrainable_variables
љ	variables
Љ	keras_api
V
њregularization_losses
Њtrainable_variables
ћ	variables
Ћ	keras_api
V
ќregularization_losses
Ќtrainable_variables
ў	variables
Ў	keras_api
n
џkernel
	Џbias
юregularization_losses
Юtrainable_variables
ъ	variables
Ъ	keras_api
а
	аaxis

Аgamma
	бbeta
Бmoving_mean
цmoving_variance
Цregularization_losses
дtrainable_variables
Д	variables
е	keras_api
V
Еregularization_losses
фtrainable_variables
Ф	variables
г	keras_api
V
Гregularization_losses
«trainable_variables
»	variables
░	keras_api
V
▒regularization_losses
▓trainable_variables
│	variables
┤	keras_api
V
хregularization_losses
Хtrainable_variables
и	variables
И	keras_api
n
╣kernel
	║bias
╗regularization_losses
╝trainable_variables
й	variables
Й	keras_api
а
	┐axis

└gamma
	┴beta
┬moving_mean
├moving_variance
─regularization_losses
┼trainable_variables
к	variables
К	keras_api
V
╚regularization_losses
╔trainable_variables
╩	variables
╦	keras_api
n
╠kernel
	═bias
╬regularization_losses
¤trainable_variables
л	variables
Л	keras_api
а
	мaxis

Мgamma
	нbeta
Нmoving_mean
оmoving_variance
Оregularization_losses
пtrainable_variables
┘	variables
┌	keras_api
V
█regularization_losses
▄trainable_variables
П	variables
я	keras_api
 
є
'0
(1
.2
/3
:4
;5
A6
B7
Q8
R9
X10
Y11
p12
q13
w14
x15
Ѓ16
ё17
і18
І19
џ20
Џ21
А22
б23
╣24
║25
└26
┴27
╠28
═29
М30
н31
ј
'0
(1
.2
/3
04
15
:6
;7
A8
B9
C10
D11
Q12
R13
X14
Y15
Z16
[17
p18
q19
w20
x21
y22
z23
Ѓ24
ё25
і26
І27
ї28
Ї29
џ30
Џ31
А32
б33
Б34
ц35
╣36
║37
└38
┴39
┬40
├41
╠42
═43
М44
н45
Н46
о47
▓
 ▀layer_regularization_losses
"regularization_losses
Яmetrics
рlayers
Рnon_trainable_variables
#trainable_variables
сlayer_metrics
$	variables
 
\Z
VARIABLE_VALUEconv2d_10/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_10/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 

'0
(1

'0
(1
▓
 Сlayer_regularization_losses
тmetrics
Тlayers
)regularization_losses
уnon_trainable_variables
*trainable_variables
Уlayer_metrics
+	variables
 
fd
VARIABLE_VALUEbatch_normalization_9/gamma5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEbatch_normalization_9/beta4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE!batch_normalization_9/moving_mean;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE%batch_normalization_9/moving_variance?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

.0
/1

.0
/1
02
13
▓
 жlayer_regularization_losses
Жmetrics
вlayers
2regularization_losses
Вnon_trainable_variables
3trainable_variables
ьlayer_metrics
4	variables
 
 
 
▓
 Ьlayer_regularization_losses
№metrics
­layers
6regularization_losses
ыnon_trainable_variables
7trainable_variables
Ыlayer_metrics
8	variables
\Z
VARIABLE_VALUEconv2d_11/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_11/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

:0
;1

:0
;1
▓
 зlayer_regularization_losses
Зmetrics
шlayers
<regularization_losses
Шnon_trainable_variables
=trainable_variables
эlayer_metrics
>	variables
 
ge
VARIABLE_VALUEbatch_normalization_10/gamma5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_10/beta4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE"batch_normalization_10/moving_mean;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE&batch_normalization_10/moving_variance?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

A0
B1

A0
B1
C2
D3
▓
 Эlayer_regularization_losses
щmetrics
Щlayers
Eregularization_losses
чnon_trainable_variables
Ftrainable_variables
Чlayer_metrics
G	variables
 
 
 
▓
 §layer_regularization_losses
■metrics
 layers
Iregularization_losses
ђnon_trainable_variables
Jtrainable_variables
Ђlayer_metrics
K	variables
 
 
 
▓
 ѓlayer_regularization_losses
Ѓmetrics
ёlayers
Mregularization_losses
Ёnon_trainable_variables
Ntrainable_variables
єlayer_metrics
O	variables
\Z
VARIABLE_VALUEconv2d_12/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_12/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE
 

Q0
R1

Q0
R1
▓
 Єlayer_regularization_losses
ѕmetrics
Ѕlayers
Sregularization_losses
іnon_trainable_variables
Ttrainable_variables
Іlayer_metrics
U	variables
 
ge
VARIABLE_VALUEbatch_normalization_11/gamma5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_11/beta4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE"batch_normalization_11/moving_mean;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE&batch_normalization_11/moving_variance?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

X0
Y1

X0
Y1
Z2
[3
▓
 їlayer_regularization_losses
Їmetrics
јlayers
\regularization_losses
Јnon_trainable_variables
]trainable_variables
љlayer_metrics
^	variables
 
 
 
▓
 Љlayer_regularization_losses
њmetrics
Њlayers
`regularization_losses
ћnon_trainable_variables
atrainable_variables
Ћlayer_metrics
b	variables
 
 
 
▓
 ќlayer_regularization_losses
Ќmetrics
ўlayers
dregularization_losses
Ўnon_trainable_variables
etrainable_variables
џlayer_metrics
f	variables
 
 
 
▓
 Џlayer_regularization_losses
юmetrics
Юlayers
hregularization_losses
ъnon_trainable_variables
itrainable_variables
Ъlayer_metrics
j	variables
 
 
 
▓
 аlayer_regularization_losses
Аmetrics
бlayers
lregularization_losses
Бnon_trainable_variables
mtrainable_variables
цlayer_metrics
n	variables
\Z
VARIABLE_VALUEconv2d_13/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_13/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE
 

p0
q1

p0
q1
▓
 Цlayer_regularization_losses
дmetrics
Дlayers
rregularization_losses
еnon_trainable_variables
strainable_variables
Еlayer_metrics
t	variables
 
ge
VARIABLE_VALUEbatch_normalization_12/gamma5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_12/beta4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE"batch_normalization_12/moving_mean;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE&batch_normalization_12/moving_variance?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

w0
x1

w0
x1
y2
z3
▓
 фlayer_regularization_losses
Фmetrics
гlayers
{regularization_losses
Гnon_trainable_variables
|trainable_variables
«layer_metrics
}	variables
 
 
 
┤
 »layer_regularization_losses
░metrics
▒layers
regularization_losses
▓non_trainable_variables
ђtrainable_variables
│layer_metrics
Ђ	variables
\Z
VARIABLE_VALUEconv2d_14/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_14/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE
 

Ѓ0
ё1

Ѓ0
ё1
х
 ┤layer_regularization_losses
хmetrics
Хlayers
Ёregularization_losses
иnon_trainable_variables
єtrainable_variables
Иlayer_metrics
Є	variables
 
ge
VARIABLE_VALUEbatch_normalization_13/gamma5layer_with_weights-9/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_13/beta4layer_with_weights-9/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE"batch_normalization_13/moving_mean;layer_with_weights-9/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE&batch_normalization_13/moving_variance?layer_with_weights-9/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

і0
І1
 
і0
І1
ї2
Ї3
х
 ╣layer_regularization_losses
║metrics
╗layers
јregularization_losses
╝non_trainable_variables
Јtrainable_variables
йlayer_metrics
љ	variables
 
 
 
х
 Йlayer_regularization_losses
┐metrics
└layers
њregularization_losses
┴non_trainable_variables
Њtrainable_variables
┬layer_metrics
ћ	variables
 
 
 
х
 ├layer_regularization_losses
─metrics
┼layers
ќregularization_losses
кnon_trainable_variables
Ќtrainable_variables
Кlayer_metrics
ў	variables
][
VARIABLE_VALUEconv2d_15/kernel7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEconv2d_15/bias5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUE
 

џ0
Џ1

џ0
Џ1
х
 ╚layer_regularization_losses
╔metrics
╩layers
юregularization_losses
╦non_trainable_variables
Юtrainable_variables
╠layer_metrics
ъ	variables
 
hf
VARIABLE_VALUEbatch_normalization_14/gamma6layer_with_weights-11/gamma/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUEbatch_normalization_14/beta5layer_with_weights-11/beta/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUE"batch_normalization_14/moving_mean<layer_with_weights-11/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUE&batch_normalization_14/moving_variance@layer_with_weights-11/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

А0
б1
 
А0
б1
Б2
ц3
х
 ═layer_regularization_losses
╬metrics
¤layers
Цregularization_losses
лnon_trainable_variables
дtrainable_variables
Лlayer_metrics
Д	variables
 
 
 
х
 мlayer_regularization_losses
Мmetrics
нlayers
Еregularization_losses
Нnon_trainable_variables
фtrainable_variables
оlayer_metrics
Ф	variables
 
 
 
х
 Оlayer_regularization_losses
пmetrics
┘layers
Гregularization_losses
┌non_trainable_variables
«trainable_variables
█layer_metrics
»	variables
 
 
 
х
 ▄layer_regularization_losses
Пmetrics
яlayers
▒regularization_losses
▀non_trainable_variables
▓trainable_variables
Яlayer_metrics
│	variables
 
 
 
х
 рlayer_regularization_losses
Рmetrics
сlayers
хregularization_losses
Сnon_trainable_variables
Хtrainable_variables
тlayer_metrics
и	variables
][
VARIABLE_VALUEconv2d_16/kernel7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEconv2d_16/bias5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUE
 

╣0
║1

╣0
║1
х
 Тlayer_regularization_losses
уmetrics
Уlayers
╗regularization_losses
жnon_trainable_variables
╝trainable_variables
Жlayer_metrics
й	variables
 
hf
VARIABLE_VALUEbatch_normalization_15/gamma6layer_with_weights-13/gamma/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUEbatch_normalization_15/beta5layer_with_weights-13/beta/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUE"batch_normalization_15/moving_mean<layer_with_weights-13/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUE&batch_normalization_15/moving_variance@layer_with_weights-13/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

└0
┴1
 
└0
┴1
┬2
├3
х
 вlayer_regularization_losses
Вmetrics
ьlayers
─regularization_losses
Ьnon_trainable_variables
┼trainable_variables
№layer_metrics
к	variables
 
 
 
х
 ­layer_regularization_losses
ыmetrics
Ыlayers
╚regularization_losses
зnon_trainable_variables
╔trainable_variables
Зlayer_metrics
╩	variables
][
VARIABLE_VALUEconv2d_17/kernel7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEconv2d_17/bias5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUE
 

╠0
═1

╠0
═1
х
 шlayer_regularization_losses
Шmetrics
эlayers
╬regularization_losses
Эnon_trainable_variables
¤trainable_variables
щlayer_metrics
л	variables
 
hf
VARIABLE_VALUEbatch_normalization_16/gamma6layer_with_weights-15/gamma/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUEbatch_normalization_16/beta5layer_with_weights-15/beta/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUE"batch_normalization_16/moving_mean<layer_with_weights-15/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUE&batch_normalization_16/moving_variance@layer_with_weights-15/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

М0
н1
 
М0
н1
Н2
о3
х
 Щlayer_regularization_losses
чmetrics
Чlayers
Оregularization_losses
§non_trainable_variables
пtrainable_variables
■layer_metrics
┘	variables
 
 
 
х
  layer_regularization_losses
ђmetrics
Ђlayers
█regularization_losses
ѓnon_trainable_variables
▄trainable_variables
Ѓlayer_metrics
П	variables
 
 
■
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
 31
!32
~
00
11
C2
D3
Z4
[5
y6
z7
ї8
Ї9
Б10
ц11
┬12
├13
Н14
о15
 
 
 
 
 
 
 
 
 

00
11
 
 
 
 
 
 
 
 
 
 
 
 
 
 

C0
D1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

Z0
[1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

y0
z1
 
 
 
 
 
 
 
 
 
 
 
 
 
 

ї0
Ї1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

Б0
ц1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

┬0
├1
 
 
 
 
 
 
 
 
 
 
 
 
 
 

Н0
о1
 
 
 
 
 
 
і
serving_default_input_3Placeholder*/
_output_shapes
:         *
dtype0*$
shape:         
┼
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_3conv2d_10/kernelconv2d_10/biasbatch_normalization_9/gammabatch_normalization_9/beta!batch_normalization_9/moving_mean%batch_normalization_9/moving_varianceconv2d_11/kernelconv2d_11/biasbatch_normalization_10/gammabatch_normalization_10/beta"batch_normalization_10/moving_mean&batch_normalization_10/moving_varianceconv2d_12/kernelconv2d_12/biasbatch_normalization_11/gammabatch_normalization_11/beta"batch_normalization_11/moving_mean&batch_normalization_11/moving_varianceconv2d_13/kernelconv2d_13/biasbatch_normalization_12/gammabatch_normalization_12/beta"batch_normalization_12/moving_mean&batch_normalization_12/moving_varianceconv2d_14/kernelconv2d_14/biasbatch_normalization_13/gammabatch_normalization_13/beta"batch_normalization_13/moving_mean&batch_normalization_13/moving_varianceconv2d_15/kernelconv2d_15/biasbatch_normalization_14/gammabatch_normalization_14/beta"batch_normalization_14/moving_mean&batch_normalization_14/moving_varianceconv2d_16/kernelconv2d_16/biasbatch_normalization_15/gammabatch_normalization_15/beta"batch_normalization_15/moving_mean&batch_normalization_15/moving_varianceconv2d_17/kernelconv2d_17/biasbatch_normalization_16/gammabatch_normalization_16/beta"batch_normalization_16/moving_mean&batch_normalization_16/moving_variance*<
Tin5
321*
Tout
2*/
_output_shapes
:         *R
_read_only_resource_inputs4
20	
 !"#$%&'()*+,-./0*3
config_proto#!

CPU

GPU2*0,1,2,3J 8*-
f(R&
$__inference_signature_wrapper_255872
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
Х
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename$conv2d_10/kernel/Read/ReadVariableOp"conv2d_10/bias/Read/ReadVariableOp/batch_normalization_9/gamma/Read/ReadVariableOp.batch_normalization_9/beta/Read/ReadVariableOp5batch_normalization_9/moving_mean/Read/ReadVariableOp9batch_normalization_9/moving_variance/Read/ReadVariableOp$conv2d_11/kernel/Read/ReadVariableOp"conv2d_11/bias/Read/ReadVariableOp0batch_normalization_10/gamma/Read/ReadVariableOp/batch_normalization_10/beta/Read/ReadVariableOp6batch_normalization_10/moving_mean/Read/ReadVariableOp:batch_normalization_10/moving_variance/Read/ReadVariableOp$conv2d_12/kernel/Read/ReadVariableOp"conv2d_12/bias/Read/ReadVariableOp0batch_normalization_11/gamma/Read/ReadVariableOp/batch_normalization_11/beta/Read/ReadVariableOp6batch_normalization_11/moving_mean/Read/ReadVariableOp:batch_normalization_11/moving_variance/Read/ReadVariableOp$conv2d_13/kernel/Read/ReadVariableOp"conv2d_13/bias/Read/ReadVariableOp0batch_normalization_12/gamma/Read/ReadVariableOp/batch_normalization_12/beta/Read/ReadVariableOp6batch_normalization_12/moving_mean/Read/ReadVariableOp:batch_normalization_12/moving_variance/Read/ReadVariableOp$conv2d_14/kernel/Read/ReadVariableOp"conv2d_14/bias/Read/ReadVariableOp0batch_normalization_13/gamma/Read/ReadVariableOp/batch_normalization_13/beta/Read/ReadVariableOp6batch_normalization_13/moving_mean/Read/ReadVariableOp:batch_normalization_13/moving_variance/Read/ReadVariableOp$conv2d_15/kernel/Read/ReadVariableOp"conv2d_15/bias/Read/ReadVariableOp0batch_normalization_14/gamma/Read/ReadVariableOp/batch_normalization_14/beta/Read/ReadVariableOp6batch_normalization_14/moving_mean/Read/ReadVariableOp:batch_normalization_14/moving_variance/Read/ReadVariableOp$conv2d_16/kernel/Read/ReadVariableOp"conv2d_16/bias/Read/ReadVariableOp0batch_normalization_15/gamma/Read/ReadVariableOp/batch_normalization_15/beta/Read/ReadVariableOp6batch_normalization_15/moving_mean/Read/ReadVariableOp:batch_normalization_15/moving_variance/Read/ReadVariableOp$conv2d_17/kernel/Read/ReadVariableOp"conv2d_17/bias/Read/ReadVariableOp0batch_normalization_16/gamma/Read/ReadVariableOp/batch_normalization_16/beta/Read/ReadVariableOp6batch_normalization_16/moving_mean/Read/ReadVariableOp:batch_normalization_16/moving_variance/Read/ReadVariableOpConst*=
Tin6
422*
Tout
2*
_output_shapes
: * 
_read_only_resource_inputs
 *3
config_proto#!

CPU

GPU2*0,1,2,3J 8*(
f#R!
__inference__traced_save_257806
ы
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_10/kernelconv2d_10/biasbatch_normalization_9/gammabatch_normalization_9/beta!batch_normalization_9/moving_mean%batch_normalization_9/moving_varianceconv2d_11/kernelconv2d_11/biasbatch_normalization_10/gammabatch_normalization_10/beta"batch_normalization_10/moving_mean&batch_normalization_10/moving_varianceconv2d_12/kernelconv2d_12/biasbatch_normalization_11/gammabatch_normalization_11/beta"batch_normalization_11/moving_mean&batch_normalization_11/moving_varianceconv2d_13/kernelconv2d_13/biasbatch_normalization_12/gammabatch_normalization_12/beta"batch_normalization_12/moving_mean&batch_normalization_12/moving_varianceconv2d_14/kernelconv2d_14/biasbatch_normalization_13/gammabatch_normalization_13/beta"batch_normalization_13/moving_mean&batch_normalization_13/moving_varianceconv2d_15/kernelconv2d_15/biasbatch_normalization_14/gammabatch_normalization_14/beta"batch_normalization_14/moving_mean&batch_normalization_14/moving_varianceconv2d_16/kernelconv2d_16/biasbatch_normalization_15/gammabatch_normalization_15/beta"batch_normalization_15/moving_mean&batch_normalization_15/moving_varianceconv2d_17/kernelconv2d_17/biasbatch_normalization_16/gammabatch_normalization_16/beta"batch_normalization_16/moving_mean&batch_normalization_16/moving_variance*<
Tin5
321*
Tout
2*
_output_shapes
: * 
_read_only_resource_inputs
 *3
config_proto#!

CPU

GPU2*0,1,2,3J 8*+
f&R$
"__inference__traced_restore_257962┐Ђ
у$
п
Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_253301

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityѕб#AssignMovingAvg/AssignSubVariableOpб%AssignMovingAvg_1/AssignSubVariableOpt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOpГ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1╔
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                            : : : : :*
epsilon%oЃ:2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
Const░
AssignMovingAvg/sub/xConst*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  ђ?2
AssignMovingAvg/sub/x┐
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const:output:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/subЦ
AssignMovingAvg/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOpя
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/sub_1К
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/mulК
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp(fusedbatchnormv3_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpХ
AssignMovingAvg_1/sub/xConst*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  ђ?2
AssignMovingAvg_1/sub/xК
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const:output:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/subФ
 AssignMovingAvg_1/ReadVariableOpReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOpЖ
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/sub_1Л
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/mulН
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp*fusedbatchnormv3_readvariableop_1_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpл
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*A
_output_shapes/
-:+                            2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                            ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
ђ	
ф
7__inference_batch_normalization_14_layer_call_fn_257409

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityѕбStatefulPartitionedCallЮ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*A
_output_shapes/
-:+                            *&
_read_only_resource_inputs
*3
config_proto#!

CPU

GPU2*0,1,2,3J 8*[
fVRT
R__inference_batch_normalization_14_layer_call_and_return_conditional_losses_2540862
StatefulPartitionedCallе
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                            2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                            ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
О
e
I__inference_activation_15_layer_call_and_return_conditional_losses_257111

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:          2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:          2

Identity"
identityIdentity:output:0*.
_input_shapes
:          :W S
/
_output_shapes
:          
 
_user_specified_nameinputs
У$
┘
R__inference_batch_normalization_16_layer_call_and_return_conditional_losses_254368

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityѕб#AssignMovingAvg/AssignSubVariableOpб%AssignMovingAvg_1/AssignSubVariableOpt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOpГ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1╔
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           :::::*
epsilon%oЃ:2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
Const░
AssignMovingAvg/sub/xConst*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  ђ?2
AssignMovingAvg/sub/x┐
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const:output:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/subЦ
AssignMovingAvg/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOpя
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:2
AssignMovingAvg/sub_1К
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:2
AssignMovingAvg/mulК
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp(fusedbatchnormv3_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpХ
AssignMovingAvg_1/sub/xConst*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  ђ?2
AssignMovingAvg_1/sub/xК
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const:output:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/subФ
 AssignMovingAvg_1/ReadVariableOpReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOpЖ
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:2
AssignMovingAvg_1/sub_1Л
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:2
AssignMovingAvg_1/mulН
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp*fusedbatchnormv3_readvariableop_1_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpл
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                           ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
╔
І
R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_256799

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityѕt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOpГ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1╩
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:          : : : : :*
epsilon%oЃ:*
is_training( 2
FusedBatchNormV3p
IdentityIdentityFusedBatchNormV3:y:0*
T0*/
_output_shapes
:          2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:          :::::W S
/
_output_shapes
:          
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
■
ф
7__inference_batch_normalization_14_layer_call_fn_257396

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityѕбStatefulPartitionedCallЏ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*A
_output_shapes/
-:+                            *$
_read_only_resource_inputs
*3
config_proto#!

CPU

GPU2*0,1,2,3J 8*[
fVRT
R__inference_batch_normalization_14_layer_call_and_return_conditional_losses_2540552
StatefulPartitionedCallе
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                            2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                            ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
а
e
I__inference_activation_19_layer_call_and_return_conditional_losses_254958

inputs
identityh
ReluReluinputs*
T0*A
_output_shapes/
-:+                            2
Reluђ
IdentityIdentityRelu:activations:0*
T0*A
_output_shapes/
-:+                            2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+                            :i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
У$
┘
R__inference_batch_normalization_12_layer_call_and_return_conditional_losses_257159

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityѕб#AssignMovingAvg/AssignSubVariableOpб%AssignMovingAvg_1/AssignSubVariableOpt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOpГ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1╔
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                            : : : : :*
epsilon%oЃ:2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
Const░
AssignMovingAvg/sub/xConst*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  ђ?2
AssignMovingAvg/sub/x┐
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const:output:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/subЦ
AssignMovingAvg/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOpя
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/sub_1К
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/mulК
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp(fusedbatchnormv3_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpХ
AssignMovingAvg_1/sub/xConst*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  ђ?2
AssignMovingAvg_1/sub/xК
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const:output:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/subФ
 AssignMovingAvg_1/ReadVariableOpReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOpЖ
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/sub_1Л
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/mulН
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp*fusedbatchnormv3_readvariableop_1_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpл
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*A
_output_shapes/
-:+                            2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                            ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
В
J
.__inference_activation_19_layer_call_fn_257441

inputs
identity╚
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*A
_output_shapes/
-:+                            * 
_read_only_resource_inputs
 *3
config_proto#!

CPU

GPU2*0,1,2,3J 8*R
fMRK
I__inference_activation_19_layer_call_and_return_conditional_losses_2549582
PartitionedCallє
IdentityIdentityPartitionedCall:output:0*
T0*A
_output_shapes/
-:+                            2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+                            :i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
ЃА
┼
C__inference_model_2_layer_call_and_return_conditional_losses_255574

inputs
conv2d_10_255445
conv2d_10_255447 
batch_normalization_9_255450 
batch_normalization_9_255452 
batch_normalization_9_255454 
batch_normalization_9_255456
conv2d_11_255460
conv2d_11_255462!
batch_normalization_10_255465!
batch_normalization_10_255467!
batch_normalization_10_255469!
batch_normalization_10_255471
conv2d_12_255476
conv2d_12_255478!
batch_normalization_11_255481!
batch_normalization_11_255483!
batch_normalization_11_255485!
batch_normalization_11_255487
conv2d_13_255494
conv2d_13_255496!
batch_normalization_12_255499!
batch_normalization_12_255501!
batch_normalization_12_255503!
batch_normalization_12_255505
conv2d_14_255509
conv2d_14_255511!
batch_normalization_13_255514!
batch_normalization_13_255516!
batch_normalization_13_255518!
batch_normalization_13_255520
conv2d_15_255525
conv2d_15_255527!
batch_normalization_14_255530!
batch_normalization_14_255532!
batch_normalization_14_255534!
batch_normalization_14_255536
conv2d_16_255543
conv2d_16_255545!
batch_normalization_15_255548!
batch_normalization_15_255550!
batch_normalization_15_255552!
batch_normalization_15_255554
conv2d_17_255558
conv2d_17_255560!
batch_normalization_16_255563!
batch_normalization_16_255565!
batch_normalization_16_255567!
batch_normalization_16_255569
identityѕб.batch_normalization_10/StatefulPartitionedCallб.batch_normalization_11/StatefulPartitionedCallб.batch_normalization_12/StatefulPartitionedCallб.batch_normalization_13/StatefulPartitionedCallб.batch_normalization_14/StatefulPartitionedCallб.batch_normalization_15/StatefulPartitionedCallб.batch_normalization_16/StatefulPartitionedCallб-batch_normalization_9/StatefulPartitionedCallб!conv2d_10/StatefulPartitionedCallб!conv2d_11/StatefulPartitionedCallб!conv2d_12/StatefulPartitionedCallб!conv2d_13/StatefulPartitionedCallб!conv2d_14/StatefulPartitionedCallб!conv2d_15/StatefulPartitionedCallб!conv2d_16/StatefulPartitionedCallб!conv2d_17/StatefulPartitionedCallѕ
!conv2d_10/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_10_255445conv2d_10_255447*
Tin
2*
Tout
2*/
_output_shapes
:          *$
_read_only_resource_inputs
*3
config_proto#!

CPU

GPU2*0,1,2,3J 8*N
fIRG
E__inference_conv2d_10_layer_call_and_return_conditional_losses_2532072#
!conv2d_10/StatefulPartitionedCallе
-batch_normalization_9/StatefulPartitionedCallStatefulPartitionedCall*conv2d_10/StatefulPartitionedCall:output:0batch_normalization_9_255450batch_normalization_9_255452batch_normalization_9_255454batch_normalization_9_255456*
Tin	
2*
Tout
2*/
_output_shapes
:          *&
_read_only_resource_inputs
*3
config_proto#!

CPU

GPU2*0,1,2,3J 8*Z
fURS
Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_2544672/
-batch_normalization_9/StatefulPartitionedCallѓ
activation_12/PartitionedCallPartitionedCall6batch_normalization_9/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:          * 
_read_only_resource_inputs
 *3
config_proto#!

CPU

GPU2*0,1,2,3J 8*R
fMRK
I__inference_activation_12_layer_call_and_return_conditional_losses_2545082
activation_12/PartitionedCallе
!conv2d_11/StatefulPartitionedCallStatefulPartitionedCall&activation_12/PartitionedCall:output:0conv2d_11_255460conv2d_11_255462*
Tin
2*
Tout
2*/
_output_shapes
:          *$
_read_only_resource_inputs
*3
config_proto#!

CPU

GPU2*0,1,2,3J 8*N
fIRG
E__inference_conv2d_11_layer_call_and_return_conditional_losses_2533542#
!conv2d_11/StatefulPartitionedCall»
.batch_normalization_10/StatefulPartitionedCallStatefulPartitionedCall*conv2d_11/StatefulPartitionedCall:output:0batch_normalization_10_255465batch_normalization_10_255467batch_normalization_10_255469batch_normalization_10_255471*
Tin	
2*
Tout
2*/
_output_shapes
:          *&
_read_only_resource_inputs
*3
config_proto#!

CPU

GPU2*0,1,2,3J 8*[
fVRT
R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_25456920
.batch_normalization_10/StatefulPartitionedCallЃ
activation_13/PartitionedCallPartitionedCall7batch_normalization_10/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:          * 
_read_only_resource_inputs
 *3
config_proto#!

CPU

GPU2*0,1,2,3J 8*R
fMRK
I__inference_activation_13_layer_call_and_return_conditional_losses_2546102
activation_13/PartitionedCallЃ
add_4/PartitionedCallPartitionedCall&activation_12/PartitionedCall:output:0&activation_13/PartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:          * 
_read_only_resource_inputs
 *3
config_proto#!

CPU

GPU2*0,1,2,3J 8*J
fERC
A__inference_add_4_layer_call_and_return_conditional_losses_2546242
add_4/PartitionedCallа
!conv2d_12/StatefulPartitionedCallStatefulPartitionedCalladd_4/PartitionedCall:output:0conv2d_12_255476conv2d_12_255478*
Tin
2*
Tout
2*/
_output_shapes
:          *$
_read_only_resource_inputs
*3
config_proto#!

CPU

GPU2*0,1,2,3J 8*N
fIRG
E__inference_conv2d_12_layer_call_and_return_conditional_losses_2535012#
!conv2d_12/StatefulPartitionedCall»
.batch_normalization_11/StatefulPartitionedCallStatefulPartitionedCall*conv2d_12/StatefulPartitionedCall:output:0batch_normalization_11_255481batch_normalization_11_255483batch_normalization_11_255485batch_normalization_11_255487*
Tin	
2*
Tout
2*/
_output_shapes
:          *&
_read_only_resource_inputs
*3
config_proto#!

CPU

GPU2*0,1,2,3J 8*[
fVRT
R__inference_batch_normalization_11_layer_call_and_return_conditional_losses_25468620
.batch_normalization_11/StatefulPartitionedCallЃ
activation_14/PartitionedCallPartitionedCall7batch_normalization_11/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:          * 
_read_only_resource_inputs
 *3
config_proto#!

CPU

GPU2*0,1,2,3J 8*R
fMRK
I__inference_activation_14_layer_call_and_return_conditional_losses_2547272
activation_14/PartitionedCallч
add_5/PartitionedCallPartitionedCalladd_4/PartitionedCall:output:0&activation_14/PartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:          * 
_read_only_resource_inputs
 *3
config_proto#!

CPU

GPU2*0,1,2,3J 8*J
fERC
A__inference_add_5_layer_call_and_return_conditional_losses_2547412
add_5/PartitionedCallЖ
activation_15/PartitionedCallPartitionedCalladd_5/PartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:          * 
_read_only_resource_inputs
 *3
config_proto#!

CPU

GPU2*0,1,2,3J 8*R
fMRK
I__inference_activation_15_layer_call_and_return_conditional_losses_2547552
activation_15/PartitionedCallі
up_sampling2d_2/PartitionedCallPartitionedCall&activation_15/PartitionedCall:output:0*
Tin
2*
Tout
2*A
_output_shapes/
-:+                            * 
_read_only_resource_inputs
 *3
config_proto#!

CPU

GPU2*0,1,2,3J 8*T
fORM
K__inference_up_sampling2d_2_layer_call_and_return_conditional_losses_2536502!
up_sampling2d_2/PartitionedCall╝
!conv2d_13/StatefulPartitionedCallStatefulPartitionedCall(up_sampling2d_2/PartitionedCall:output:0conv2d_13_255494conv2d_13_255496*
Tin
2*
Tout
2*A
_output_shapes/
-:+                            *$
_read_only_resource_inputs
*3
config_proto#!

CPU

GPU2*0,1,2,3J 8*N
fIRG
E__inference_conv2d_13_layer_call_and_return_conditional_losses_2536672#
!conv2d_13/StatefulPartitionedCall┴
.batch_normalization_12/StatefulPartitionedCallStatefulPartitionedCall*conv2d_13/StatefulPartitionedCall:output:0batch_normalization_12_255499batch_normalization_12_255501batch_normalization_12_255503batch_normalization_12_255505*
Tin	
2*
Tout
2*A
_output_shapes/
-:+                            *&
_read_only_resource_inputs
*3
config_proto#!

CPU

GPU2*0,1,2,3J 8*[
fVRT
R__inference_batch_normalization_12_layer_call_and_return_conditional_losses_25379220
.batch_normalization_12/StatefulPartitionedCallЋ
activation_16/PartitionedCallPartitionedCall7batch_normalization_12/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*A
_output_shapes/
-:+                            * 
_read_only_resource_inputs
 *3
config_proto#!

CPU

GPU2*0,1,2,3J 8*R
fMRK
I__inference_activation_16_layer_call_and_return_conditional_losses_2548092
activation_16/PartitionedCall║
!conv2d_14/StatefulPartitionedCallStatefulPartitionedCall&activation_16/PartitionedCall:output:0conv2d_14_255509conv2d_14_255511*
Tin
2*
Tout
2*A
_output_shapes/
-:+                            *$
_read_only_resource_inputs
*3
config_proto#!

CPU

GPU2*0,1,2,3J 8*N
fIRG
E__inference_conv2d_14_layer_call_and_return_conditional_losses_2538142#
!conv2d_14/StatefulPartitionedCall┴
.batch_normalization_13/StatefulPartitionedCallStatefulPartitionedCall*conv2d_14/StatefulPartitionedCall:output:0batch_normalization_13_255514batch_normalization_13_255516batch_normalization_13_255518batch_normalization_13_255520*
Tin	
2*
Tout
2*A
_output_shapes/
-:+                            *&
_read_only_resource_inputs
*3
config_proto#!

CPU

GPU2*0,1,2,3J 8*[
fVRT
R__inference_batch_normalization_13_layer_call_and_return_conditional_losses_25393920
.batch_normalization_13/StatefulPartitionedCallЋ
activation_17/PartitionedCallPartitionedCall7batch_normalization_13/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*A
_output_shapes/
-:+                            * 
_read_only_resource_inputs
 *3
config_proto#!

CPU

GPU2*0,1,2,3J 8*R
fMRK
I__inference_activation_17_layer_call_and_return_conditional_losses_2548622
activation_17/PartitionedCallЋ
add_6/PartitionedCallPartitionedCall&activation_16/PartitionedCall:output:0&activation_17/PartitionedCall:output:0*
Tin
2*
Tout
2*A
_output_shapes/
-:+                            * 
_read_only_resource_inputs
 *3
config_proto#!

CPU

GPU2*0,1,2,3J 8*J
fERC
A__inference_add_6_layer_call_and_return_conditional_losses_2548762
add_6/PartitionedCall▓
!conv2d_15/StatefulPartitionedCallStatefulPartitionedCalladd_6/PartitionedCall:output:0conv2d_15_255525conv2d_15_255527*
Tin
2*
Tout
2*A
_output_shapes/
-:+                            *$
_read_only_resource_inputs
*3
config_proto#!

CPU

GPU2*0,1,2,3J 8*N
fIRG
E__inference_conv2d_15_layer_call_and_return_conditional_losses_2539612#
!conv2d_15/StatefulPartitionedCall┴
.batch_normalization_14/StatefulPartitionedCallStatefulPartitionedCall*conv2d_15/StatefulPartitionedCall:output:0batch_normalization_14_255530batch_normalization_14_255532batch_normalization_14_255534batch_normalization_14_255536*
Tin	
2*
Tout
2*A
_output_shapes/
-:+                            *&
_read_only_resource_inputs
*3
config_proto#!

CPU

GPU2*0,1,2,3J 8*[
fVRT
R__inference_batch_normalization_14_layer_call_and_return_conditional_losses_25408620
.batch_normalization_14/StatefulPartitionedCallЋ
activation_18/PartitionedCallPartitionedCall7batch_normalization_14/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*A
_output_shapes/
-:+                            * 
_read_only_resource_inputs
 *3
config_proto#!

CPU

GPU2*0,1,2,3J 8*R
fMRK
I__inference_activation_18_layer_call_and_return_conditional_losses_2549302
activation_18/PartitionedCallЇ
add_7/PartitionedCallPartitionedCalladd_6/PartitionedCall:output:0&activation_18/PartitionedCall:output:0*
Tin
2*
Tout
2*A
_output_shapes/
-:+                            * 
_read_only_resource_inputs
 *3
config_proto#!

CPU

GPU2*0,1,2,3J 8*J
fERC
A__inference_add_7_layer_call_and_return_conditional_losses_2549442
add_7/PartitionedCallЧ
activation_19/PartitionedCallPartitionedCalladd_7/PartitionedCall:output:0*
Tin
2*
Tout
2*A
_output_shapes/
-:+                            * 
_read_only_resource_inputs
 *3
config_proto#!

CPU

GPU2*0,1,2,3J 8*R
fMRK
I__inference_activation_19_layer_call_and_return_conditional_losses_2549582
activation_19/PartitionedCallі
up_sampling2d_3/PartitionedCallPartitionedCall&activation_19/PartitionedCall:output:0*
Tin
2*
Tout
2*A
_output_shapes/
-:+                            * 
_read_only_resource_inputs
 *3
config_proto#!

CPU

GPU2*0,1,2,3J 8*T
fORM
K__inference_up_sampling2d_3_layer_call_and_return_conditional_losses_2541102!
up_sampling2d_3/PartitionedCall╝
!conv2d_16/StatefulPartitionedCallStatefulPartitionedCall(up_sampling2d_3/PartitionedCall:output:0conv2d_16_255543conv2d_16_255545*
Tin
2*
Tout
2*A
_output_shapes/
-:+                            *$
_read_only_resource_inputs
*3
config_proto#!

CPU

GPU2*0,1,2,3J 8*N
fIRG
E__inference_conv2d_16_layer_call_and_return_conditional_losses_2541272#
!conv2d_16/StatefulPartitionedCall┴
.batch_normalization_15/StatefulPartitionedCallStatefulPartitionedCall*conv2d_16/StatefulPartitionedCall:output:0batch_normalization_15_255548batch_normalization_15_255550batch_normalization_15_255552batch_normalization_15_255554*
Tin	
2*
Tout
2*A
_output_shapes/
-:+                            *&
_read_only_resource_inputs
*3
config_proto#!

CPU

GPU2*0,1,2,3J 8*[
fVRT
R__inference_batch_normalization_15_layer_call_and_return_conditional_losses_25425220
.batch_normalization_15/StatefulPartitionedCallЋ
activation_20/PartitionedCallPartitionedCall7batch_normalization_15/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*A
_output_shapes/
-:+                            * 
_read_only_resource_inputs
 *3
config_proto#!

CPU

GPU2*0,1,2,3J 8*R
fMRK
I__inference_activation_20_layer_call_and_return_conditional_losses_2550122
activation_20/PartitionedCall║
!conv2d_17/StatefulPartitionedCallStatefulPartitionedCall&activation_20/PartitionedCall:output:0conv2d_17_255558conv2d_17_255560*
Tin
2*
Tout
2*A
_output_shapes/
-:+                           *$
_read_only_resource_inputs
*3
config_proto#!

CPU

GPU2*0,1,2,3J 8*N
fIRG
E__inference_conv2d_17_layer_call_and_return_conditional_losses_2542742#
!conv2d_17/StatefulPartitionedCall┴
.batch_normalization_16/StatefulPartitionedCallStatefulPartitionedCall*conv2d_17/StatefulPartitionedCall:output:0batch_normalization_16_255563batch_normalization_16_255565batch_normalization_16_255567batch_normalization_16_255569*
Tin	
2*
Tout
2*A
_output_shapes/
-:+                           *&
_read_only_resource_inputs
*3
config_proto#!

CPU

GPU2*0,1,2,3J 8*[
fVRT
R__inference_batch_normalization_16_layer_call_and_return_conditional_losses_25439920
.batch_normalization_16/StatefulPartitionedCallЋ
activation_21/PartitionedCallPartitionedCall7batch_normalization_16/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*A
_output_shapes/
-:+                           * 
_read_only_resource_inputs
 *3
config_proto#!

CPU

GPU2*0,1,2,3J 8*R
fMRK
I__inference_activation_21_layer_call_and_return_conditional_losses_2550652
activation_21/PartitionedCall╗
IdentityIdentity&activation_21/PartitionedCall:output:0/^batch_normalization_10/StatefulPartitionedCall/^batch_normalization_11/StatefulPartitionedCall/^batch_normalization_12/StatefulPartitionedCall/^batch_normalization_13/StatefulPartitionedCall/^batch_normalization_14/StatefulPartitionedCall/^batch_normalization_15/StatefulPartitionedCall/^batch_normalization_16/StatefulPartitionedCall.^batch_normalization_9/StatefulPartitionedCall"^conv2d_10/StatefulPartitionedCall"^conv2d_11/StatefulPartitionedCall"^conv2d_12/StatefulPartitionedCall"^conv2d_13/StatefulPartitionedCall"^conv2d_14/StatefulPartitionedCall"^conv2d_15/StatefulPartitionedCall"^conv2d_16/StatefulPartitionedCall"^conv2d_17/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*­
_input_shapesя
█:         ::::::::::::::::::::::::::::::::::::::::::::::::2`
.batch_normalization_10/StatefulPartitionedCall.batch_normalization_10/StatefulPartitionedCall2`
.batch_normalization_11/StatefulPartitionedCall.batch_normalization_11/StatefulPartitionedCall2`
.batch_normalization_12/StatefulPartitionedCall.batch_normalization_12/StatefulPartitionedCall2`
.batch_normalization_13/StatefulPartitionedCall.batch_normalization_13/StatefulPartitionedCall2`
.batch_normalization_14/StatefulPartitionedCall.batch_normalization_14/StatefulPartitionedCall2`
.batch_normalization_15/StatefulPartitionedCall.batch_normalization_15/StatefulPartitionedCall2`
.batch_normalization_16/StatefulPartitionedCall.batch_normalization_16/StatefulPartitionedCall2^
-batch_normalization_9/StatefulPartitionedCall-batch_normalization_9/StatefulPartitionedCall2F
!conv2d_10/StatefulPartitionedCall!conv2d_10/StatefulPartitionedCall2F
!conv2d_11/StatefulPartitionedCall!conv2d_11/StatefulPartitionedCall2F
!conv2d_12/StatefulPartitionedCall!conv2d_12/StatefulPartitionedCall2F
!conv2d_13/StatefulPartitionedCall!conv2d_13/StatefulPartitionedCall2F
!conv2d_14/StatefulPartitionedCall!conv2d_14/StatefulPartitionedCall2F
!conv2d_15/StatefulPartitionedCall!conv2d_15/StatefulPartitionedCall2F
!conv2d_16/StatefulPartitionedCall!conv2d_16/StatefulPartitionedCall2F
!conv2d_17/StatefulPartitionedCall!conv2d_17/StatefulPartitionedCall:W S
/
_output_shapes
:         
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
: :!

_output_shapes
: :"

_output_shapes
: :#

_output_shapes
: :$

_output_shapes
: :%

_output_shapes
: :&

_output_shapes
: :'

_output_shapes
: :(

_output_shapes
: :)

_output_shapes
: :*

_output_shapes
: :+

_output_shapes
: :,

_output_shapes
: :-

_output_shapes
: :.

_output_shapes
: :/

_output_shapes
: :0

_output_shapes
: 
╝	
Г
E__inference_conv2d_14_layer_call_and_return_conditional_losses_253814

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityѕЋ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02
Conv2D/ReadVariableOpх
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                            *
paddingSAME*
strides
2
Conv2Dї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOpџ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                            2	
BiasAdd~
IdentityIdentityBiasAdd:output:0*
T0*A
_output_shapes/
-:+                            2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+                            :::i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
њ
І
R__inference_batch_normalization_14_layer_call_and_return_conditional_losses_257383

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityѕt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOpГ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1▄
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                            : : : : :*
epsilon%oЃ:*
is_training( 2
FusedBatchNormV3ѓ
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+                            2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                            :::::i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
В
J
.__inference_activation_17_layer_call_fn_257310

inputs
identity╚
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*A
_output_shapes/
-:+                            * 
_read_only_resource_inputs
 *3
config_proto#!

CPU

GPU2*0,1,2,3J 8*R
fMRK
I__inference_activation_17_layer_call_and_return_conditional_losses_2548622
PartitionedCallє
IdentityIdentityPartitionedCall:output:0*
T0*A
_output_shapes/
-:+                            2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+                            :i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
ж

*__inference_conv2d_14_layer_call_fn_253824

inputs
unknown
	unknown_0
identityѕбStatefulPartitionedCallШ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*A
_output_shapes/
-:+                            *$
_read_only_resource_inputs
*3
config_proto#!

CPU

GPU2*0,1,2,3J 8*N
fIRG
E__inference_conv2d_14_layer_call_and_return_conditional_losses_2538142
StatefulPartitionedCallе
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                            2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+                            ::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
У$
┘
R__inference_batch_normalization_15_layer_call_and_return_conditional_losses_257484

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityѕб#AssignMovingAvg/AssignSubVariableOpб%AssignMovingAvg_1/AssignSubVariableOpt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOpГ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1╔
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                            : : : : :*
epsilon%oЃ:2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
Const░
AssignMovingAvg/sub/xConst*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  ђ?2
AssignMovingAvg/sub/x┐
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const:output:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/subЦ
AssignMovingAvg/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOpя
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/sub_1К
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/mulК
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp(fusedbatchnormv3_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpХ
AssignMovingAvg_1/sub/xConst*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  ђ?2
AssignMovingAvg_1/sub/xК
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const:output:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/subФ
 AssignMovingAvg_1/ReadVariableOpReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOpЖ
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/sub_1Л
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/mulН
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp*fusedbatchnormv3_readvariableop_1_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpл
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*A
_output_shapes/
-:+                            2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                            ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
■
ф
7__inference_batch_normalization_11_layer_call_fn_257071

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityѕбStatefulPartitionedCallЏ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*A
_output_shapes/
-:+                            *$
_read_only_resource_inputs
*3
config_proto#!

CPU

GPU2*0,1,2,3J 8*[
fVRT
R__inference_batch_normalization_11_layer_call_and_return_conditional_losses_2535952
StatefulPartitionedCallе
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                            2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                            ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
О
e
I__inference_activation_15_layer_call_and_return_conditional_losses_254755

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:          2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:          2

Identity"
identityIdentity:output:0*.
_input_shapes
:          :W S
/
_output_shapes
:          
 
_user_specified_nameinputs
╚
і
Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_254467

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityѕt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOpГ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1╩
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:          : : : : :*
epsilon%oЃ:*
is_training( 2
FusedBatchNormV3p
IdentityIdentityFusedBatchNormV3:y:0*
T0*/
_output_shapes
:          2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:          :::::W S
/
_output_shapes
:          
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
¤
k
A__inference_add_4_layer_call_and_return_conditional_losses_254624

inputs
inputs_1
identity_
addAddV2inputsinputs_1*
T0*/
_output_shapes
:          2
addc
IdentityIdentityadd:z:0*
T0*/
_output_shapes
:          2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:          :          :W S
/
_output_shapes
:          
 
_user_specified_nameinputs:WS
/
_output_shapes
:          
 
_user_specified_nameinputs
╝	
Г
E__inference_conv2d_13_layer_call_and_return_conditional_losses_253667

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityѕЋ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02
Conv2D/ReadVariableOpх
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                            *
paddingSAME*
strides
2
Conv2Dї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOpџ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                            2	
BiasAdd~
IdentityIdentityBiasAdd:output:0*
T0*A
_output_shapes/
-:+                            2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+                            :::i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
ћ
L
0__inference_up_sampling2d_3_layer_call_fn_254116

inputs
identityМ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*J
_output_shapes8
6:4                                    * 
_read_only_resource_inputs
 *3
config_proto#!

CPU

GPU2*0,1,2,3J 8*T
fORM
K__inference_up_sampling2d_3_layer_call_and_return_conditional_losses_2541102
PartitionedCallЈ
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4                                    2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
ж

*__inference_conv2d_16_layer_call_fn_254137

inputs
unknown
	unknown_0
identityѕбStatefulPartitionedCallШ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*A
_output_shapes/
-:+                            *$
_read_only_resource_inputs
*3
config_proto#!

CPU

GPU2*0,1,2,3J 8*N
fIRG
E__inference_conv2d_16_layer_call_and_return_conditional_losses_2541272
StatefulPartitionedCallе
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                            2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+                            ::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
Љ
і
Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_253332

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityѕt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOpГ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1▄
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                            : : : : :*
epsilon%oЃ:*
is_training( 2
FusedBatchNormV3ѓ
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+                            2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                            :::::i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
њ
І
R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_253479

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityѕt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOpГ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1▄
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                            : : : : :*
epsilon%oЃ:*
is_training( 2
FusedBatchNormV3ѓ
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+                            2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                            :::::i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
а
e
I__inference_activation_17_layer_call_and_return_conditional_losses_254862

inputs
identityh
ReluReluinputs*
T0*A
_output_shapes/
-:+                            2
Reluђ
IdentityIdentityRelu:activations:0*
T0*A
_output_shapes/
-:+                            2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+                            :i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
ђ	
ф
7__inference_batch_normalization_10_layer_call_fn_256900

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityѕбStatefulPartitionedCallЮ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*A
_output_shapes/
-:+                            *&
_read_only_resource_inputs
*3
config_proto#!

CPU

GPU2*0,1,2,3J 8*[
fVRT
R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_2534792
StatefulPartitionedCallе
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                            2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                            ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
■
ф
7__inference_batch_normalization_13_layer_call_fn_257287

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityѕбStatefulPartitionedCallЏ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*A
_output_shapes/
-:+                            *$
_read_only_resource_inputs
*3
config_proto#!

CPU

GPU2*0,1,2,3J 8*[
fVRT
R__inference_batch_normalization_13_layer_call_and_return_conditional_losses_2539082
StatefulPartitionedCallе
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                            2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                            ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
У$
┘
R__inference_batch_normalization_13_layer_call_and_return_conditional_losses_257256

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityѕб#AssignMovingAvg/AssignSubVariableOpб%AssignMovingAvg_1/AssignSubVariableOpt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOpГ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1╔
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                            : : : : :*
epsilon%oЃ:2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
Const░
AssignMovingAvg/sub/xConst*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  ђ?2
AssignMovingAvg/sub/x┐
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const:output:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/subЦ
AssignMovingAvg/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOpя
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/sub_1К
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/mulК
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp(fusedbatchnormv3_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpХ
AssignMovingAvg_1/sub/xConst*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  ђ?2
AssignMovingAvg_1/sub/xК
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const:output:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/subФ
 AssignMovingAvg_1/ReadVariableOpReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOpЖ
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/sub_1Л
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/mulН
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp*fusedbatchnormv3_readvariableop_1_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpл
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*A
_output_shapes/
-:+                            2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                            ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
єА
к
C__inference_model_2_layer_call_and_return_conditional_losses_255206
input_3
conv2d_10_255077
conv2d_10_255079 
batch_normalization_9_255082 
batch_normalization_9_255084 
batch_normalization_9_255086 
batch_normalization_9_255088
conv2d_11_255092
conv2d_11_255094!
batch_normalization_10_255097!
batch_normalization_10_255099!
batch_normalization_10_255101!
batch_normalization_10_255103
conv2d_12_255108
conv2d_12_255110!
batch_normalization_11_255113!
batch_normalization_11_255115!
batch_normalization_11_255117!
batch_normalization_11_255119
conv2d_13_255126
conv2d_13_255128!
batch_normalization_12_255131!
batch_normalization_12_255133!
batch_normalization_12_255135!
batch_normalization_12_255137
conv2d_14_255141
conv2d_14_255143!
batch_normalization_13_255146!
batch_normalization_13_255148!
batch_normalization_13_255150!
batch_normalization_13_255152
conv2d_15_255157
conv2d_15_255159!
batch_normalization_14_255162!
batch_normalization_14_255164!
batch_normalization_14_255166!
batch_normalization_14_255168
conv2d_16_255175
conv2d_16_255177!
batch_normalization_15_255180!
batch_normalization_15_255182!
batch_normalization_15_255184!
batch_normalization_15_255186
conv2d_17_255190
conv2d_17_255192!
batch_normalization_16_255195!
batch_normalization_16_255197!
batch_normalization_16_255199!
batch_normalization_16_255201
identityѕб.batch_normalization_10/StatefulPartitionedCallб.batch_normalization_11/StatefulPartitionedCallб.batch_normalization_12/StatefulPartitionedCallб.batch_normalization_13/StatefulPartitionedCallб.batch_normalization_14/StatefulPartitionedCallб.batch_normalization_15/StatefulPartitionedCallб.batch_normalization_16/StatefulPartitionedCallб-batch_normalization_9/StatefulPartitionedCallб!conv2d_10/StatefulPartitionedCallб!conv2d_11/StatefulPartitionedCallб!conv2d_12/StatefulPartitionedCallб!conv2d_13/StatefulPartitionedCallб!conv2d_14/StatefulPartitionedCallб!conv2d_15/StatefulPartitionedCallб!conv2d_16/StatefulPartitionedCallб!conv2d_17/StatefulPartitionedCallЅ
!conv2d_10/StatefulPartitionedCallStatefulPartitionedCallinput_3conv2d_10_255077conv2d_10_255079*
Tin
2*
Tout
2*/
_output_shapes
:          *$
_read_only_resource_inputs
*3
config_proto#!

CPU

GPU2*0,1,2,3J 8*N
fIRG
E__inference_conv2d_10_layer_call_and_return_conditional_losses_2532072#
!conv2d_10/StatefulPartitionedCallе
-batch_normalization_9/StatefulPartitionedCallStatefulPartitionedCall*conv2d_10/StatefulPartitionedCall:output:0batch_normalization_9_255082batch_normalization_9_255084batch_normalization_9_255086batch_normalization_9_255088*
Tin	
2*
Tout
2*/
_output_shapes
:          *&
_read_only_resource_inputs
*3
config_proto#!

CPU

GPU2*0,1,2,3J 8*Z
fURS
Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_2544672/
-batch_normalization_9/StatefulPartitionedCallѓ
activation_12/PartitionedCallPartitionedCall6batch_normalization_9/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:          * 
_read_only_resource_inputs
 *3
config_proto#!

CPU

GPU2*0,1,2,3J 8*R
fMRK
I__inference_activation_12_layer_call_and_return_conditional_losses_2545082
activation_12/PartitionedCallе
!conv2d_11/StatefulPartitionedCallStatefulPartitionedCall&activation_12/PartitionedCall:output:0conv2d_11_255092conv2d_11_255094*
Tin
2*
Tout
2*/
_output_shapes
:          *$
_read_only_resource_inputs
*3
config_proto#!

CPU

GPU2*0,1,2,3J 8*N
fIRG
E__inference_conv2d_11_layer_call_and_return_conditional_losses_2533542#
!conv2d_11/StatefulPartitionedCall»
.batch_normalization_10/StatefulPartitionedCallStatefulPartitionedCall*conv2d_11/StatefulPartitionedCall:output:0batch_normalization_10_255097batch_normalization_10_255099batch_normalization_10_255101batch_normalization_10_255103*
Tin	
2*
Tout
2*/
_output_shapes
:          *&
_read_only_resource_inputs
*3
config_proto#!

CPU

GPU2*0,1,2,3J 8*[
fVRT
R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_25456920
.batch_normalization_10/StatefulPartitionedCallЃ
activation_13/PartitionedCallPartitionedCall7batch_normalization_10/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:          * 
_read_only_resource_inputs
 *3
config_proto#!

CPU

GPU2*0,1,2,3J 8*R
fMRK
I__inference_activation_13_layer_call_and_return_conditional_losses_2546102
activation_13/PartitionedCallЃ
add_4/PartitionedCallPartitionedCall&activation_12/PartitionedCall:output:0&activation_13/PartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:          * 
_read_only_resource_inputs
 *3
config_proto#!

CPU

GPU2*0,1,2,3J 8*J
fERC
A__inference_add_4_layer_call_and_return_conditional_losses_2546242
add_4/PartitionedCallа
!conv2d_12/StatefulPartitionedCallStatefulPartitionedCalladd_4/PartitionedCall:output:0conv2d_12_255108conv2d_12_255110*
Tin
2*
Tout
2*/
_output_shapes
:          *$
_read_only_resource_inputs
*3
config_proto#!

CPU

GPU2*0,1,2,3J 8*N
fIRG
E__inference_conv2d_12_layer_call_and_return_conditional_losses_2535012#
!conv2d_12/StatefulPartitionedCall»
.batch_normalization_11/StatefulPartitionedCallStatefulPartitionedCall*conv2d_12/StatefulPartitionedCall:output:0batch_normalization_11_255113batch_normalization_11_255115batch_normalization_11_255117batch_normalization_11_255119*
Tin	
2*
Tout
2*/
_output_shapes
:          *&
_read_only_resource_inputs
*3
config_proto#!

CPU

GPU2*0,1,2,3J 8*[
fVRT
R__inference_batch_normalization_11_layer_call_and_return_conditional_losses_25468620
.batch_normalization_11/StatefulPartitionedCallЃ
activation_14/PartitionedCallPartitionedCall7batch_normalization_11/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:          * 
_read_only_resource_inputs
 *3
config_proto#!

CPU

GPU2*0,1,2,3J 8*R
fMRK
I__inference_activation_14_layer_call_and_return_conditional_losses_2547272
activation_14/PartitionedCallч
add_5/PartitionedCallPartitionedCalladd_4/PartitionedCall:output:0&activation_14/PartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:          * 
_read_only_resource_inputs
 *3
config_proto#!

CPU

GPU2*0,1,2,3J 8*J
fERC
A__inference_add_5_layer_call_and_return_conditional_losses_2547412
add_5/PartitionedCallЖ
activation_15/PartitionedCallPartitionedCalladd_5/PartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:          * 
_read_only_resource_inputs
 *3
config_proto#!

CPU

GPU2*0,1,2,3J 8*R
fMRK
I__inference_activation_15_layer_call_and_return_conditional_losses_2547552
activation_15/PartitionedCallі
up_sampling2d_2/PartitionedCallPartitionedCall&activation_15/PartitionedCall:output:0*
Tin
2*
Tout
2*A
_output_shapes/
-:+                            * 
_read_only_resource_inputs
 *3
config_proto#!

CPU

GPU2*0,1,2,3J 8*T
fORM
K__inference_up_sampling2d_2_layer_call_and_return_conditional_losses_2536502!
up_sampling2d_2/PartitionedCall╝
!conv2d_13/StatefulPartitionedCallStatefulPartitionedCall(up_sampling2d_2/PartitionedCall:output:0conv2d_13_255126conv2d_13_255128*
Tin
2*
Tout
2*A
_output_shapes/
-:+                            *$
_read_only_resource_inputs
*3
config_proto#!

CPU

GPU2*0,1,2,3J 8*N
fIRG
E__inference_conv2d_13_layer_call_and_return_conditional_losses_2536672#
!conv2d_13/StatefulPartitionedCall┴
.batch_normalization_12/StatefulPartitionedCallStatefulPartitionedCall*conv2d_13/StatefulPartitionedCall:output:0batch_normalization_12_255131batch_normalization_12_255133batch_normalization_12_255135batch_normalization_12_255137*
Tin	
2*
Tout
2*A
_output_shapes/
-:+                            *&
_read_only_resource_inputs
*3
config_proto#!

CPU

GPU2*0,1,2,3J 8*[
fVRT
R__inference_batch_normalization_12_layer_call_and_return_conditional_losses_25379220
.batch_normalization_12/StatefulPartitionedCallЋ
activation_16/PartitionedCallPartitionedCall7batch_normalization_12/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*A
_output_shapes/
-:+                            * 
_read_only_resource_inputs
 *3
config_proto#!

CPU

GPU2*0,1,2,3J 8*R
fMRK
I__inference_activation_16_layer_call_and_return_conditional_losses_2548092
activation_16/PartitionedCall║
!conv2d_14/StatefulPartitionedCallStatefulPartitionedCall&activation_16/PartitionedCall:output:0conv2d_14_255141conv2d_14_255143*
Tin
2*
Tout
2*A
_output_shapes/
-:+                            *$
_read_only_resource_inputs
*3
config_proto#!

CPU

GPU2*0,1,2,3J 8*N
fIRG
E__inference_conv2d_14_layer_call_and_return_conditional_losses_2538142#
!conv2d_14/StatefulPartitionedCall┴
.batch_normalization_13/StatefulPartitionedCallStatefulPartitionedCall*conv2d_14/StatefulPartitionedCall:output:0batch_normalization_13_255146batch_normalization_13_255148batch_normalization_13_255150batch_normalization_13_255152*
Tin	
2*
Tout
2*A
_output_shapes/
-:+                            *&
_read_only_resource_inputs
*3
config_proto#!

CPU

GPU2*0,1,2,3J 8*[
fVRT
R__inference_batch_normalization_13_layer_call_and_return_conditional_losses_25393920
.batch_normalization_13/StatefulPartitionedCallЋ
activation_17/PartitionedCallPartitionedCall7batch_normalization_13/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*A
_output_shapes/
-:+                            * 
_read_only_resource_inputs
 *3
config_proto#!

CPU

GPU2*0,1,2,3J 8*R
fMRK
I__inference_activation_17_layer_call_and_return_conditional_losses_2548622
activation_17/PartitionedCallЋ
add_6/PartitionedCallPartitionedCall&activation_16/PartitionedCall:output:0&activation_17/PartitionedCall:output:0*
Tin
2*
Tout
2*A
_output_shapes/
-:+                            * 
_read_only_resource_inputs
 *3
config_proto#!

CPU

GPU2*0,1,2,3J 8*J
fERC
A__inference_add_6_layer_call_and_return_conditional_losses_2548762
add_6/PartitionedCall▓
!conv2d_15/StatefulPartitionedCallStatefulPartitionedCalladd_6/PartitionedCall:output:0conv2d_15_255157conv2d_15_255159*
Tin
2*
Tout
2*A
_output_shapes/
-:+                            *$
_read_only_resource_inputs
*3
config_proto#!

CPU

GPU2*0,1,2,3J 8*N
fIRG
E__inference_conv2d_15_layer_call_and_return_conditional_losses_2539612#
!conv2d_15/StatefulPartitionedCall┴
.batch_normalization_14/StatefulPartitionedCallStatefulPartitionedCall*conv2d_15/StatefulPartitionedCall:output:0batch_normalization_14_255162batch_normalization_14_255164batch_normalization_14_255166batch_normalization_14_255168*
Tin	
2*
Tout
2*A
_output_shapes/
-:+                            *&
_read_only_resource_inputs
*3
config_proto#!

CPU

GPU2*0,1,2,3J 8*[
fVRT
R__inference_batch_normalization_14_layer_call_and_return_conditional_losses_25408620
.batch_normalization_14/StatefulPartitionedCallЋ
activation_18/PartitionedCallPartitionedCall7batch_normalization_14/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*A
_output_shapes/
-:+                            * 
_read_only_resource_inputs
 *3
config_proto#!

CPU

GPU2*0,1,2,3J 8*R
fMRK
I__inference_activation_18_layer_call_and_return_conditional_losses_2549302
activation_18/PartitionedCallЇ
add_7/PartitionedCallPartitionedCalladd_6/PartitionedCall:output:0&activation_18/PartitionedCall:output:0*
Tin
2*
Tout
2*A
_output_shapes/
-:+                            * 
_read_only_resource_inputs
 *3
config_proto#!

CPU

GPU2*0,1,2,3J 8*J
fERC
A__inference_add_7_layer_call_and_return_conditional_losses_2549442
add_7/PartitionedCallЧ
activation_19/PartitionedCallPartitionedCalladd_7/PartitionedCall:output:0*
Tin
2*
Tout
2*A
_output_shapes/
-:+                            * 
_read_only_resource_inputs
 *3
config_proto#!

CPU

GPU2*0,1,2,3J 8*R
fMRK
I__inference_activation_19_layer_call_and_return_conditional_losses_2549582
activation_19/PartitionedCallі
up_sampling2d_3/PartitionedCallPartitionedCall&activation_19/PartitionedCall:output:0*
Tin
2*
Tout
2*A
_output_shapes/
-:+                            * 
_read_only_resource_inputs
 *3
config_proto#!

CPU

GPU2*0,1,2,3J 8*T
fORM
K__inference_up_sampling2d_3_layer_call_and_return_conditional_losses_2541102!
up_sampling2d_3/PartitionedCall╝
!conv2d_16/StatefulPartitionedCallStatefulPartitionedCall(up_sampling2d_3/PartitionedCall:output:0conv2d_16_255175conv2d_16_255177*
Tin
2*
Tout
2*A
_output_shapes/
-:+                            *$
_read_only_resource_inputs
*3
config_proto#!

CPU

GPU2*0,1,2,3J 8*N
fIRG
E__inference_conv2d_16_layer_call_and_return_conditional_losses_2541272#
!conv2d_16/StatefulPartitionedCall┴
.batch_normalization_15/StatefulPartitionedCallStatefulPartitionedCall*conv2d_16/StatefulPartitionedCall:output:0batch_normalization_15_255180batch_normalization_15_255182batch_normalization_15_255184batch_normalization_15_255186*
Tin	
2*
Tout
2*A
_output_shapes/
-:+                            *&
_read_only_resource_inputs
*3
config_proto#!

CPU

GPU2*0,1,2,3J 8*[
fVRT
R__inference_batch_normalization_15_layer_call_and_return_conditional_losses_25425220
.batch_normalization_15/StatefulPartitionedCallЋ
activation_20/PartitionedCallPartitionedCall7batch_normalization_15/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*A
_output_shapes/
-:+                            * 
_read_only_resource_inputs
 *3
config_proto#!

CPU

GPU2*0,1,2,3J 8*R
fMRK
I__inference_activation_20_layer_call_and_return_conditional_losses_2550122
activation_20/PartitionedCall║
!conv2d_17/StatefulPartitionedCallStatefulPartitionedCall&activation_20/PartitionedCall:output:0conv2d_17_255190conv2d_17_255192*
Tin
2*
Tout
2*A
_output_shapes/
-:+                           *$
_read_only_resource_inputs
*3
config_proto#!

CPU

GPU2*0,1,2,3J 8*N
fIRG
E__inference_conv2d_17_layer_call_and_return_conditional_losses_2542742#
!conv2d_17/StatefulPartitionedCall┴
.batch_normalization_16/StatefulPartitionedCallStatefulPartitionedCall*conv2d_17/StatefulPartitionedCall:output:0batch_normalization_16_255195batch_normalization_16_255197batch_normalization_16_255199batch_normalization_16_255201*
Tin	
2*
Tout
2*A
_output_shapes/
-:+                           *&
_read_only_resource_inputs
*3
config_proto#!

CPU

GPU2*0,1,2,3J 8*[
fVRT
R__inference_batch_normalization_16_layer_call_and_return_conditional_losses_25439920
.batch_normalization_16/StatefulPartitionedCallЋ
activation_21/PartitionedCallPartitionedCall7batch_normalization_16/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*A
_output_shapes/
-:+                           * 
_read_only_resource_inputs
 *3
config_proto#!

CPU

GPU2*0,1,2,3J 8*R
fMRK
I__inference_activation_21_layer_call_and_return_conditional_losses_2550652
activation_21/PartitionedCall╗
IdentityIdentity&activation_21/PartitionedCall:output:0/^batch_normalization_10/StatefulPartitionedCall/^batch_normalization_11/StatefulPartitionedCall/^batch_normalization_12/StatefulPartitionedCall/^batch_normalization_13/StatefulPartitionedCall/^batch_normalization_14/StatefulPartitionedCall/^batch_normalization_15/StatefulPartitionedCall/^batch_normalization_16/StatefulPartitionedCall.^batch_normalization_9/StatefulPartitionedCall"^conv2d_10/StatefulPartitionedCall"^conv2d_11/StatefulPartitionedCall"^conv2d_12/StatefulPartitionedCall"^conv2d_13/StatefulPartitionedCall"^conv2d_14/StatefulPartitionedCall"^conv2d_15/StatefulPartitionedCall"^conv2d_16/StatefulPartitionedCall"^conv2d_17/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*­
_input_shapesя
█:         ::::::::::::::::::::::::::::::::::::::::::::::::2`
.batch_normalization_10/StatefulPartitionedCall.batch_normalization_10/StatefulPartitionedCall2`
.batch_normalization_11/StatefulPartitionedCall.batch_normalization_11/StatefulPartitionedCall2`
.batch_normalization_12/StatefulPartitionedCall.batch_normalization_12/StatefulPartitionedCall2`
.batch_normalization_13/StatefulPartitionedCall.batch_normalization_13/StatefulPartitionedCall2`
.batch_normalization_14/StatefulPartitionedCall.batch_normalization_14/StatefulPartitionedCall2`
.batch_normalization_15/StatefulPartitionedCall.batch_normalization_15/StatefulPartitionedCall2`
.batch_normalization_16/StatefulPartitionedCall.batch_normalization_16/StatefulPartitionedCall2^
-batch_normalization_9/StatefulPartitionedCall-batch_normalization_9/StatefulPartitionedCall2F
!conv2d_10/StatefulPartitionedCall!conv2d_10/StatefulPartitionedCall2F
!conv2d_11/StatefulPartitionedCall!conv2d_11/StatefulPartitionedCall2F
!conv2d_12/StatefulPartitionedCall!conv2d_12/StatefulPartitionedCall2F
!conv2d_13/StatefulPartitionedCall!conv2d_13/StatefulPartitionedCall2F
!conv2d_14/StatefulPartitionedCall!conv2d_14/StatefulPartitionedCall2F
!conv2d_15/StatefulPartitionedCall!conv2d_15/StatefulPartitionedCall2F
!conv2d_16/StatefulPartitionedCall!conv2d_16/StatefulPartitionedCall2F
!conv2d_17/StatefulPartitionedCall!conv2d_17/StatefulPartitionedCall:X T
/
_output_shapes
:         
!
_user_specified_name	input_3:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
: :!

_output_shapes
: :"

_output_shapes
: :#

_output_shapes
: :$

_output_shapes
: :%

_output_shapes
: :&

_output_shapes
: :'

_output_shapes
: :(

_output_shapes
: :)

_output_shapes
: :*

_output_shapes
: :+

_output_shapes
: :,

_output_shapes
: :-

_output_shapes
: :.

_output_shapes
: :/

_output_shapes
: :0

_output_shapes
: 
┤
Е
6__inference_batch_normalization_9_layer_call_fn_256715

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityѕбStatefulPartitionedCallѕ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*/
_output_shapes
:          *$
_read_only_resource_inputs
*3
config_proto#!

CPU

GPU2*0,1,2,3J 8*Z
fURS
Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_2544492
StatefulPartitionedCallќ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:          2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:          ::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:          
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Ъ$
п
Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_254449

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityѕб#AssignMovingAvg/AssignSubVariableOpб%AssignMovingAvg_1/AssignSubVariableOpt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOpГ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1и
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:          : : : : :*
epsilon%oЃ:2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
Const░
AssignMovingAvg/sub/xConst*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  ђ?2
AssignMovingAvg/sub/x┐
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const:output:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/subЦ
AssignMovingAvg/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOpя
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/sub_1К
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/mulК
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp(fusedbatchnormv3_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpХ
AssignMovingAvg_1/sub/xConst*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  ђ?2
AssignMovingAvg_1/sub/xК
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const:output:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/subФ
 AssignMovingAvg_1/ReadVariableOpReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOpЖ
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/sub_1Л
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/mulН
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp*fusedbatchnormv3_readvariableop_1_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpЙ
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*/
_output_shapes
:          2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:          ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:W S
/
_output_shapes
:          
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
╝	
Г
E__inference_conv2d_12_layer_call_and_return_conditional_losses_253501

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityѕЋ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02
Conv2D/ReadVariableOpх
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                            *
paddingSAME*
strides
2
Conv2Dї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOpџ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                            2	
BiasAdd~
IdentityIdentityBiasAdd:output:0*
T0*A
_output_shapes/
-:+                            2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+                            :::i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
О
m
A__inference_add_4_layer_call_and_return_conditional_losses_256916
inputs_0
inputs_1
identitya
addAddV2inputs_0inputs_1*
T0*/
_output_shapes
:          2
addc
IdentityIdentityadd:z:0*
T0*/
_output_shapes
:          2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:          :          :Y U
/
_output_shapes
:          
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:          
"
_user_specified_name
inputs/1
О
e
I__inference_activation_13_layer_call_and_return_conditional_losses_256905

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:          2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:          2

Identity"
identityIdentity:output:0*.
_input_shapes
:          :W S
/
_output_shapes
:          
 
_user_specified_nameinputs
њ
І
R__inference_batch_normalization_11_layer_call_and_return_conditional_losses_257058

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityѕt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOpГ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1▄
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                            : : : : :*
epsilon%oЃ:*
is_training( 2
FusedBatchNormV3ѓ
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+                            2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                            :::::i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Х
Е
6__inference_batch_normalization_9_layer_call_fn_256728

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityѕбStatefulPartitionedCallі
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*/
_output_shapes
:          *&
_read_only_resource_inputs
*3
config_proto#!

CPU

GPU2*0,1,2,3J 8*Z
fURS
Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_2544672
StatefulPartitionedCallќ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:          2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:          ::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:          
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
О
e
I__inference_activation_13_layer_call_and_return_conditional_losses_254610

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:          2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:          2

Identity"
identityIdentity:output:0*.
_input_shapes
:          :W S
/
_output_shapes
:          
 
_user_specified_nameinputs
╔
І
R__inference_batch_normalization_11_layer_call_and_return_conditional_losses_256983

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityѕt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOpГ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1╩
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:          : : : : :*
epsilon%oЃ:*
is_training( 2
FusedBatchNormV3p
IdentityIdentityFusedBatchNormV3:y:0*
T0*/
_output_shapes
:          2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:          :::::W S
/
_output_shapes
:          
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
ѕ
g
K__inference_up_sampling2d_2_layer_call_and_return_conditional_losses_253650

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2╬
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
strided_slice_
ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
Const^
mulMulstrided_slice:output:0Const:output:0*
T0*
_output_shapes
:2
mulН
resize/ResizeNearestNeighborResizeNearestNeighborinputsmul:z:0*
T0*J
_output_shapes8
6:4                                    *
half_pixel_centers(2
resize/ResizeNearestNeighborц
IdentityIdentity-resize/ResizeNearestNeighbor:resized_images:0*
T0*J
_output_shapes8
6:4                                    2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
┼в
н
C__inference_model_2_layer_call_and_return_conditional_losses_256170

inputs,
(conv2d_10_conv2d_readvariableop_resource-
)conv2d_10_biasadd_readvariableop_resource1
-batch_normalization_9_readvariableop_resource3
/batch_normalization_9_readvariableop_1_resourceB
>batch_normalization_9_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource,
(conv2d_11_conv2d_readvariableop_resource-
)conv2d_11_biasadd_readvariableop_resource2
.batch_normalization_10_readvariableop_resource4
0batch_normalization_10_readvariableop_1_resourceC
?batch_normalization_10_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_10_fusedbatchnormv3_readvariableop_1_resource,
(conv2d_12_conv2d_readvariableop_resource-
)conv2d_12_biasadd_readvariableop_resource2
.batch_normalization_11_readvariableop_resource4
0batch_normalization_11_readvariableop_1_resourceC
?batch_normalization_11_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_11_fusedbatchnormv3_readvariableop_1_resource,
(conv2d_13_conv2d_readvariableop_resource-
)conv2d_13_biasadd_readvariableop_resource2
.batch_normalization_12_readvariableop_resource4
0batch_normalization_12_readvariableop_1_resourceC
?batch_normalization_12_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_12_fusedbatchnormv3_readvariableop_1_resource,
(conv2d_14_conv2d_readvariableop_resource-
)conv2d_14_biasadd_readvariableop_resource2
.batch_normalization_13_readvariableop_resource4
0batch_normalization_13_readvariableop_1_resourceC
?batch_normalization_13_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_13_fusedbatchnormv3_readvariableop_1_resource,
(conv2d_15_conv2d_readvariableop_resource-
)conv2d_15_biasadd_readvariableop_resource2
.batch_normalization_14_readvariableop_resource4
0batch_normalization_14_readvariableop_1_resourceC
?batch_normalization_14_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_14_fusedbatchnormv3_readvariableop_1_resource,
(conv2d_16_conv2d_readvariableop_resource-
)conv2d_16_biasadd_readvariableop_resource2
.batch_normalization_15_readvariableop_resource4
0batch_normalization_15_readvariableop_1_resourceC
?batch_normalization_15_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_15_fusedbatchnormv3_readvariableop_1_resource,
(conv2d_17_conv2d_readvariableop_resource-
)conv2d_17_biasadd_readvariableop_resource2
.batch_normalization_16_readvariableop_resource4
0batch_normalization_16_readvariableop_1_resourceC
?batch_normalization_16_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_16_fusedbatchnormv3_readvariableop_1_resource
identityѕб:batch_normalization_10/AssignMovingAvg/AssignSubVariableOpб<batch_normalization_10/AssignMovingAvg_1/AssignSubVariableOpб:batch_normalization_11/AssignMovingAvg/AssignSubVariableOpб<batch_normalization_11/AssignMovingAvg_1/AssignSubVariableOpб:batch_normalization_12/AssignMovingAvg/AssignSubVariableOpб<batch_normalization_12/AssignMovingAvg_1/AssignSubVariableOpб:batch_normalization_13/AssignMovingAvg/AssignSubVariableOpб<batch_normalization_13/AssignMovingAvg_1/AssignSubVariableOpб:batch_normalization_14/AssignMovingAvg/AssignSubVariableOpб<batch_normalization_14/AssignMovingAvg_1/AssignSubVariableOpб:batch_normalization_15/AssignMovingAvg/AssignSubVariableOpб<batch_normalization_15/AssignMovingAvg_1/AssignSubVariableOpб:batch_normalization_16/AssignMovingAvg/AssignSubVariableOpб<batch_normalization_16/AssignMovingAvg_1/AssignSubVariableOpб9batch_normalization_9/AssignMovingAvg/AssignSubVariableOpб;batch_normalization_9/AssignMovingAvg_1/AssignSubVariableOp│
conv2d_10/Conv2D/ReadVariableOpReadVariableOp(conv2d_10_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_10/Conv2D/ReadVariableOp┴
conv2d_10/Conv2DConv2Dinputs'conv2d_10/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:          *
paddingSAME*
strides
2
conv2d_10/Conv2Dф
 conv2d_10/BiasAdd/ReadVariableOpReadVariableOp)conv2d_10_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_10/BiasAdd/ReadVariableOp░
conv2d_10/BiasAddBiasAddconv2d_10/Conv2D:output:0(conv2d_10/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          2
conv2d_10/BiasAddХ
$batch_normalization_9/ReadVariableOpReadVariableOp-batch_normalization_9_readvariableop_resource*
_output_shapes
: *
dtype02&
$batch_normalization_9/ReadVariableOp╝
&batch_normalization_9/ReadVariableOp_1ReadVariableOp/batch_normalization_9_readvariableop_1_resource*
_output_shapes
: *
dtype02(
&batch_normalization_9/ReadVariableOp_1ж
5batch_normalization_9/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_9_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype027
5batch_normalization_9/FusedBatchNormV3/ReadVariableOp№
7batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype029
7batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1¤
&batch_normalization_9/FusedBatchNormV3FusedBatchNormV3conv2d_10/BiasAdd:output:0,batch_normalization_9/ReadVariableOp:value:0.batch_normalization_9/ReadVariableOp_1:value:0=batch_normalization_9/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:          : : : : :*
epsilon%oЃ:2(
&batch_normalization_9/FusedBatchNormV3
batch_normalization_9/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
batch_normalization_9/ConstЫ
+batch_normalization_9/AssignMovingAvg/sub/xConst*Q
_classG
ECloc:@batch_normalization_9/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  ђ?2-
+batch_normalization_9/AssignMovingAvg/sub/xГ
)batch_normalization_9/AssignMovingAvg/subSub4batch_normalization_9/AssignMovingAvg/sub/x:output:0$batch_normalization_9/Const:output:0*
T0*Q
_classG
ECloc:@batch_normalization_9/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2+
)batch_normalization_9/AssignMovingAvg/subу
4batch_normalization_9/AssignMovingAvg/ReadVariableOpReadVariableOp>batch_normalization_9_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype026
4batch_normalization_9/AssignMovingAvg/ReadVariableOp╠
+batch_normalization_9/AssignMovingAvg/sub_1Sub<batch_normalization_9/AssignMovingAvg/ReadVariableOp:value:03batch_normalization_9/FusedBatchNormV3:batch_mean:0*
T0*Q
_classG
ECloc:@batch_normalization_9/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2-
+batch_normalization_9/AssignMovingAvg/sub_1х
)batch_normalization_9/AssignMovingAvg/mulMul/batch_normalization_9/AssignMovingAvg/sub_1:z:0-batch_normalization_9/AssignMovingAvg/sub:z:0*
T0*Q
_classG
ECloc:@batch_normalization_9/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2+
)batch_normalization_9/AssignMovingAvg/mulр
9batch_normalization_9/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp>batch_normalization_9_fusedbatchnormv3_readvariableop_resource-batch_normalization_9/AssignMovingAvg/mul:z:05^batch_normalization_9/AssignMovingAvg/ReadVariableOp6^batch_normalization_9/FusedBatchNormV3/ReadVariableOp*Q
_classG
ECloc:@batch_normalization_9/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02;
9batch_normalization_9/AssignMovingAvg/AssignSubVariableOpЭ
-batch_normalization_9/AssignMovingAvg_1/sub/xConst*S
_classI
GEloc:@batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  ђ?2/
-batch_normalization_9/AssignMovingAvg_1/sub/xх
+batch_normalization_9/AssignMovingAvg_1/subSub6batch_normalization_9/AssignMovingAvg_1/sub/x:output:0$batch_normalization_9/Const:output:0*
T0*S
_classI
GEloc:@batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2-
+batch_normalization_9/AssignMovingAvg_1/subь
6batch_normalization_9/AssignMovingAvg_1/ReadVariableOpReadVariableOp@batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype028
6batch_normalization_9/AssignMovingAvg_1/ReadVariableOpп
-batch_normalization_9/AssignMovingAvg_1/sub_1Sub>batch_normalization_9/AssignMovingAvg_1/ReadVariableOp:value:07batch_normalization_9/FusedBatchNormV3:batch_variance:0*
T0*S
_classI
GEloc:@batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2/
-batch_normalization_9/AssignMovingAvg_1/sub_1┐
+batch_normalization_9/AssignMovingAvg_1/mulMul1batch_normalization_9/AssignMovingAvg_1/sub_1:z:0/batch_normalization_9/AssignMovingAvg_1/sub:z:0*
T0*S
_classI
GEloc:@batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2-
+batch_normalization_9/AssignMovingAvg_1/mul№
;batch_normalization_9/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp@batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource/batch_normalization_9/AssignMovingAvg_1/mul:z:07^batch_normalization_9/AssignMovingAvg_1/ReadVariableOp8^batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1*S
_classI
GEloc:@batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02=
;batch_normalization_9/AssignMovingAvg_1/AssignSubVariableOpќ
activation_12/ReluRelu*batch_normalization_9/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:          2
activation_12/Relu│
conv2d_11/Conv2D/ReadVariableOpReadVariableOp(conv2d_11_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02!
conv2d_11/Conv2D/ReadVariableOp█
conv2d_11/Conv2DConv2D activation_12/Relu:activations:0'conv2d_11/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:          *
paddingSAME*
strides
2
conv2d_11/Conv2Dф
 conv2d_11/BiasAdd/ReadVariableOpReadVariableOp)conv2d_11_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_11/BiasAdd/ReadVariableOp░
conv2d_11/BiasAddBiasAddconv2d_11/Conv2D:output:0(conv2d_11/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          2
conv2d_11/BiasAdd╣
%batch_normalization_10/ReadVariableOpReadVariableOp.batch_normalization_10_readvariableop_resource*
_output_shapes
: *
dtype02'
%batch_normalization_10/ReadVariableOp┐
'batch_normalization_10/ReadVariableOp_1ReadVariableOp0batch_normalization_10_readvariableop_1_resource*
_output_shapes
: *
dtype02)
'batch_normalization_10/ReadVariableOp_1В
6batch_normalization_10/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_10_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype028
6batch_normalization_10/FusedBatchNormV3/ReadVariableOpЫ
8batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_10_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02:
8batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1Н
'batch_normalization_10/FusedBatchNormV3FusedBatchNormV3conv2d_11/BiasAdd:output:0-batch_normalization_10/ReadVariableOp:value:0/batch_normalization_10/ReadVariableOp_1:value:0>batch_normalization_10/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:          : : : : :*
epsilon%oЃ:2)
'batch_normalization_10/FusedBatchNormV3Ђ
batch_normalization_10/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
batch_normalization_10/Constш
,batch_normalization_10/AssignMovingAvg/sub/xConst*R
_classH
FDloc:@batch_normalization_10/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  ђ?2.
,batch_normalization_10/AssignMovingAvg/sub/x▓
*batch_normalization_10/AssignMovingAvg/subSub5batch_normalization_10/AssignMovingAvg/sub/x:output:0%batch_normalization_10/Const:output:0*
T0*R
_classH
FDloc:@batch_normalization_10/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2,
*batch_normalization_10/AssignMovingAvg/subЖ
5batch_normalization_10/AssignMovingAvg/ReadVariableOpReadVariableOp?batch_normalization_10_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype027
5batch_normalization_10/AssignMovingAvg/ReadVariableOpЛ
,batch_normalization_10/AssignMovingAvg/sub_1Sub=batch_normalization_10/AssignMovingAvg/ReadVariableOp:value:04batch_normalization_10/FusedBatchNormV3:batch_mean:0*
T0*R
_classH
FDloc:@batch_normalization_10/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2.
,batch_normalization_10/AssignMovingAvg/sub_1║
*batch_normalization_10/AssignMovingAvg/mulMul0batch_normalization_10/AssignMovingAvg/sub_1:z:0.batch_normalization_10/AssignMovingAvg/sub:z:0*
T0*R
_classH
FDloc:@batch_normalization_10/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2,
*batch_normalization_10/AssignMovingAvg/mulУ
:batch_normalization_10/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp?batch_normalization_10_fusedbatchnormv3_readvariableop_resource.batch_normalization_10/AssignMovingAvg/mul:z:06^batch_normalization_10/AssignMovingAvg/ReadVariableOp7^batch_normalization_10/FusedBatchNormV3/ReadVariableOp*R
_classH
FDloc:@batch_normalization_10/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02<
:batch_normalization_10/AssignMovingAvg/AssignSubVariableOpч
.batch_normalization_10/AssignMovingAvg_1/sub/xConst*T
_classJ
HFloc:@batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  ђ?20
.batch_normalization_10/AssignMovingAvg_1/sub/x║
,batch_normalization_10/AssignMovingAvg_1/subSub7batch_normalization_10/AssignMovingAvg_1/sub/x:output:0%batch_normalization_10/Const:output:0*
T0*T
_classJ
HFloc:@batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2.
,batch_normalization_10/AssignMovingAvg_1/sub­
7batch_normalization_10/AssignMovingAvg_1/ReadVariableOpReadVariableOpAbatch_normalization_10_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype029
7batch_normalization_10/AssignMovingAvg_1/ReadVariableOpП
.batch_normalization_10/AssignMovingAvg_1/sub_1Sub?batch_normalization_10/AssignMovingAvg_1/ReadVariableOp:value:08batch_normalization_10/FusedBatchNormV3:batch_variance:0*
T0*T
_classJ
HFloc:@batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 20
.batch_normalization_10/AssignMovingAvg_1/sub_1─
,batch_normalization_10/AssignMovingAvg_1/mulMul2batch_normalization_10/AssignMovingAvg_1/sub_1:z:00batch_normalization_10/AssignMovingAvg_1/sub:z:0*
T0*T
_classJ
HFloc:@batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2.
,batch_normalization_10/AssignMovingAvg_1/mulШ
<batch_normalization_10/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpAbatch_normalization_10_fusedbatchnormv3_readvariableop_1_resource0batch_normalization_10/AssignMovingAvg_1/mul:z:08^batch_normalization_10/AssignMovingAvg_1/ReadVariableOp9^batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1*T
_classJ
HFloc:@batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02>
<batch_normalization_10/AssignMovingAvg_1/AssignSubVariableOpЌ
activation_13/ReluRelu+batch_normalization_10/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:          2
activation_13/ReluЮ
	add_4/addAddV2 activation_12/Relu:activations:0 activation_13/Relu:activations:0*
T0*/
_output_shapes
:          2
	add_4/add│
conv2d_12/Conv2D/ReadVariableOpReadVariableOp(conv2d_12_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02!
conv2d_12/Conv2D/ReadVariableOp╚
conv2d_12/Conv2DConv2Dadd_4/add:z:0'conv2d_12/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:          *
paddingSAME*
strides
2
conv2d_12/Conv2Dф
 conv2d_12/BiasAdd/ReadVariableOpReadVariableOp)conv2d_12_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_12/BiasAdd/ReadVariableOp░
conv2d_12/BiasAddBiasAddconv2d_12/Conv2D:output:0(conv2d_12/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          2
conv2d_12/BiasAdd╣
%batch_normalization_11/ReadVariableOpReadVariableOp.batch_normalization_11_readvariableop_resource*
_output_shapes
: *
dtype02'
%batch_normalization_11/ReadVariableOp┐
'batch_normalization_11/ReadVariableOp_1ReadVariableOp0batch_normalization_11_readvariableop_1_resource*
_output_shapes
: *
dtype02)
'batch_normalization_11/ReadVariableOp_1В
6batch_normalization_11/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_11_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype028
6batch_normalization_11/FusedBatchNormV3/ReadVariableOpЫ
8batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_11_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02:
8batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1Н
'batch_normalization_11/FusedBatchNormV3FusedBatchNormV3conv2d_12/BiasAdd:output:0-batch_normalization_11/ReadVariableOp:value:0/batch_normalization_11/ReadVariableOp_1:value:0>batch_normalization_11/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:          : : : : :*
epsilon%oЃ:2)
'batch_normalization_11/FusedBatchNormV3Ђ
batch_normalization_11/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
batch_normalization_11/Constш
,batch_normalization_11/AssignMovingAvg/sub/xConst*R
_classH
FDloc:@batch_normalization_11/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  ђ?2.
,batch_normalization_11/AssignMovingAvg/sub/x▓
*batch_normalization_11/AssignMovingAvg/subSub5batch_normalization_11/AssignMovingAvg/sub/x:output:0%batch_normalization_11/Const:output:0*
T0*R
_classH
FDloc:@batch_normalization_11/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2,
*batch_normalization_11/AssignMovingAvg/subЖ
5batch_normalization_11/AssignMovingAvg/ReadVariableOpReadVariableOp?batch_normalization_11_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype027
5batch_normalization_11/AssignMovingAvg/ReadVariableOpЛ
,batch_normalization_11/AssignMovingAvg/sub_1Sub=batch_normalization_11/AssignMovingAvg/ReadVariableOp:value:04batch_normalization_11/FusedBatchNormV3:batch_mean:0*
T0*R
_classH
FDloc:@batch_normalization_11/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2.
,batch_normalization_11/AssignMovingAvg/sub_1║
*batch_normalization_11/AssignMovingAvg/mulMul0batch_normalization_11/AssignMovingAvg/sub_1:z:0.batch_normalization_11/AssignMovingAvg/sub:z:0*
T0*R
_classH
FDloc:@batch_normalization_11/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2,
*batch_normalization_11/AssignMovingAvg/mulУ
:batch_normalization_11/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp?batch_normalization_11_fusedbatchnormv3_readvariableop_resource.batch_normalization_11/AssignMovingAvg/mul:z:06^batch_normalization_11/AssignMovingAvg/ReadVariableOp7^batch_normalization_11/FusedBatchNormV3/ReadVariableOp*R
_classH
FDloc:@batch_normalization_11/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02<
:batch_normalization_11/AssignMovingAvg/AssignSubVariableOpч
.batch_normalization_11/AssignMovingAvg_1/sub/xConst*T
_classJ
HFloc:@batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  ђ?20
.batch_normalization_11/AssignMovingAvg_1/sub/x║
,batch_normalization_11/AssignMovingAvg_1/subSub7batch_normalization_11/AssignMovingAvg_1/sub/x:output:0%batch_normalization_11/Const:output:0*
T0*T
_classJ
HFloc:@batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2.
,batch_normalization_11/AssignMovingAvg_1/sub­
7batch_normalization_11/AssignMovingAvg_1/ReadVariableOpReadVariableOpAbatch_normalization_11_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype029
7batch_normalization_11/AssignMovingAvg_1/ReadVariableOpП
.batch_normalization_11/AssignMovingAvg_1/sub_1Sub?batch_normalization_11/AssignMovingAvg_1/ReadVariableOp:value:08batch_normalization_11/FusedBatchNormV3:batch_variance:0*
T0*T
_classJ
HFloc:@batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 20
.batch_normalization_11/AssignMovingAvg_1/sub_1─
,batch_normalization_11/AssignMovingAvg_1/mulMul2batch_normalization_11/AssignMovingAvg_1/sub_1:z:00batch_normalization_11/AssignMovingAvg_1/sub:z:0*
T0*T
_classJ
HFloc:@batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2.
,batch_normalization_11/AssignMovingAvg_1/mulШ
<batch_normalization_11/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpAbatch_normalization_11_fusedbatchnormv3_readvariableop_1_resource0batch_normalization_11/AssignMovingAvg_1/mul:z:08^batch_normalization_11/AssignMovingAvg_1/ReadVariableOp9^batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1*T
_classJ
HFloc:@batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02>
<batch_normalization_11/AssignMovingAvg_1/AssignSubVariableOpЌ
activation_14/ReluRelu+batch_normalization_11/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:          2
activation_14/Reluі
	add_5/addAddV2add_4/add:z:0 activation_14/Relu:activations:0*
T0*/
_output_shapes
:          2
	add_5/addy
activation_15/ReluReluadd_5/add:z:0*
T0*/
_output_shapes
:          2
activation_15/Relu~
up_sampling2d_2/ShapeShape activation_15/Relu:activations:0*
T0*
_output_shapes
:2
up_sampling2d_2/Shapeћ
#up_sampling2d_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2%
#up_sampling2d_2/strided_slice/stackў
%up_sampling2d_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%up_sampling2d_2/strided_slice/stack_1ў
%up_sampling2d_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%up_sampling2d_2/strided_slice/stack_2«
up_sampling2d_2/strided_sliceStridedSliceup_sampling2d_2/Shape:output:0,up_sampling2d_2/strided_slice/stack:output:0.up_sampling2d_2/strided_slice/stack_1:output:0.up_sampling2d_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
up_sampling2d_2/strided_slice
up_sampling2d_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d_2/Constъ
up_sampling2d_2/mulMul&up_sampling2d_2/strided_slice:output:0up_sampling2d_2/Const:output:0*
T0*
_output_shapes
:2
up_sampling2d_2/mulё
,up_sampling2d_2/resize/ResizeNearestNeighborResizeNearestNeighbor activation_15/Relu:activations:0up_sampling2d_2/mul:z:0*
T0*/
_output_shapes
:          *
half_pixel_centers(2.
,up_sampling2d_2/resize/ResizeNearestNeighbor│
conv2d_13/Conv2D/ReadVariableOpReadVariableOp(conv2d_13_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02!
conv2d_13/Conv2D/ReadVariableOpЭ
conv2d_13/Conv2DConv2D=up_sampling2d_2/resize/ResizeNearestNeighbor:resized_images:0'conv2d_13/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:          *
paddingSAME*
strides
2
conv2d_13/Conv2Dф
 conv2d_13/BiasAdd/ReadVariableOpReadVariableOp)conv2d_13_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_13/BiasAdd/ReadVariableOp░
conv2d_13/BiasAddBiasAddconv2d_13/Conv2D:output:0(conv2d_13/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          2
conv2d_13/BiasAdd╣
%batch_normalization_12/ReadVariableOpReadVariableOp.batch_normalization_12_readvariableop_resource*
_output_shapes
: *
dtype02'
%batch_normalization_12/ReadVariableOp┐
'batch_normalization_12/ReadVariableOp_1ReadVariableOp0batch_normalization_12_readvariableop_1_resource*
_output_shapes
: *
dtype02)
'batch_normalization_12/ReadVariableOp_1В
6batch_normalization_12/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_12_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype028
6batch_normalization_12/FusedBatchNormV3/ReadVariableOpЫ
8batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_12_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02:
8batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1Н
'batch_normalization_12/FusedBatchNormV3FusedBatchNormV3conv2d_13/BiasAdd:output:0-batch_normalization_12/ReadVariableOp:value:0/batch_normalization_12/ReadVariableOp_1:value:0>batch_normalization_12/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:          : : : : :*
epsilon%oЃ:2)
'batch_normalization_12/FusedBatchNormV3Ђ
batch_normalization_12/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
batch_normalization_12/Constш
,batch_normalization_12/AssignMovingAvg/sub/xConst*R
_classH
FDloc:@batch_normalization_12/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  ђ?2.
,batch_normalization_12/AssignMovingAvg/sub/x▓
*batch_normalization_12/AssignMovingAvg/subSub5batch_normalization_12/AssignMovingAvg/sub/x:output:0%batch_normalization_12/Const:output:0*
T0*R
_classH
FDloc:@batch_normalization_12/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2,
*batch_normalization_12/AssignMovingAvg/subЖ
5batch_normalization_12/AssignMovingAvg/ReadVariableOpReadVariableOp?batch_normalization_12_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype027
5batch_normalization_12/AssignMovingAvg/ReadVariableOpЛ
,batch_normalization_12/AssignMovingAvg/sub_1Sub=batch_normalization_12/AssignMovingAvg/ReadVariableOp:value:04batch_normalization_12/FusedBatchNormV3:batch_mean:0*
T0*R
_classH
FDloc:@batch_normalization_12/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2.
,batch_normalization_12/AssignMovingAvg/sub_1║
*batch_normalization_12/AssignMovingAvg/mulMul0batch_normalization_12/AssignMovingAvg/sub_1:z:0.batch_normalization_12/AssignMovingAvg/sub:z:0*
T0*R
_classH
FDloc:@batch_normalization_12/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2,
*batch_normalization_12/AssignMovingAvg/mulУ
:batch_normalization_12/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp?batch_normalization_12_fusedbatchnormv3_readvariableop_resource.batch_normalization_12/AssignMovingAvg/mul:z:06^batch_normalization_12/AssignMovingAvg/ReadVariableOp7^batch_normalization_12/FusedBatchNormV3/ReadVariableOp*R
_classH
FDloc:@batch_normalization_12/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02<
:batch_normalization_12/AssignMovingAvg/AssignSubVariableOpч
.batch_normalization_12/AssignMovingAvg_1/sub/xConst*T
_classJ
HFloc:@batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  ђ?20
.batch_normalization_12/AssignMovingAvg_1/sub/x║
,batch_normalization_12/AssignMovingAvg_1/subSub7batch_normalization_12/AssignMovingAvg_1/sub/x:output:0%batch_normalization_12/Const:output:0*
T0*T
_classJ
HFloc:@batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2.
,batch_normalization_12/AssignMovingAvg_1/sub­
7batch_normalization_12/AssignMovingAvg_1/ReadVariableOpReadVariableOpAbatch_normalization_12_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype029
7batch_normalization_12/AssignMovingAvg_1/ReadVariableOpП
.batch_normalization_12/AssignMovingAvg_1/sub_1Sub?batch_normalization_12/AssignMovingAvg_1/ReadVariableOp:value:08batch_normalization_12/FusedBatchNormV3:batch_variance:0*
T0*T
_classJ
HFloc:@batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 20
.batch_normalization_12/AssignMovingAvg_1/sub_1─
,batch_normalization_12/AssignMovingAvg_1/mulMul2batch_normalization_12/AssignMovingAvg_1/sub_1:z:00batch_normalization_12/AssignMovingAvg_1/sub:z:0*
T0*T
_classJ
HFloc:@batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2.
,batch_normalization_12/AssignMovingAvg_1/mulШ
<batch_normalization_12/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpAbatch_normalization_12_fusedbatchnormv3_readvariableop_1_resource0batch_normalization_12/AssignMovingAvg_1/mul:z:08^batch_normalization_12/AssignMovingAvg_1/ReadVariableOp9^batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1*T
_classJ
HFloc:@batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02>
<batch_normalization_12/AssignMovingAvg_1/AssignSubVariableOpЌ
activation_16/ReluRelu+batch_normalization_12/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:          2
activation_16/Relu│
conv2d_14/Conv2D/ReadVariableOpReadVariableOp(conv2d_14_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02!
conv2d_14/Conv2D/ReadVariableOp█
conv2d_14/Conv2DConv2D activation_16/Relu:activations:0'conv2d_14/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:          *
paddingSAME*
strides
2
conv2d_14/Conv2Dф
 conv2d_14/BiasAdd/ReadVariableOpReadVariableOp)conv2d_14_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_14/BiasAdd/ReadVariableOp░
conv2d_14/BiasAddBiasAddconv2d_14/Conv2D:output:0(conv2d_14/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          2
conv2d_14/BiasAdd╣
%batch_normalization_13/ReadVariableOpReadVariableOp.batch_normalization_13_readvariableop_resource*
_output_shapes
: *
dtype02'
%batch_normalization_13/ReadVariableOp┐
'batch_normalization_13/ReadVariableOp_1ReadVariableOp0batch_normalization_13_readvariableop_1_resource*
_output_shapes
: *
dtype02)
'batch_normalization_13/ReadVariableOp_1В
6batch_normalization_13/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_13_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype028
6batch_normalization_13/FusedBatchNormV3/ReadVariableOpЫ
8batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_13_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02:
8batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1Н
'batch_normalization_13/FusedBatchNormV3FusedBatchNormV3conv2d_14/BiasAdd:output:0-batch_normalization_13/ReadVariableOp:value:0/batch_normalization_13/ReadVariableOp_1:value:0>batch_normalization_13/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:          : : : : :*
epsilon%oЃ:2)
'batch_normalization_13/FusedBatchNormV3Ђ
batch_normalization_13/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
batch_normalization_13/Constш
,batch_normalization_13/AssignMovingAvg/sub/xConst*R
_classH
FDloc:@batch_normalization_13/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  ђ?2.
,batch_normalization_13/AssignMovingAvg/sub/x▓
*batch_normalization_13/AssignMovingAvg/subSub5batch_normalization_13/AssignMovingAvg/sub/x:output:0%batch_normalization_13/Const:output:0*
T0*R
_classH
FDloc:@batch_normalization_13/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2,
*batch_normalization_13/AssignMovingAvg/subЖ
5batch_normalization_13/AssignMovingAvg/ReadVariableOpReadVariableOp?batch_normalization_13_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype027
5batch_normalization_13/AssignMovingAvg/ReadVariableOpЛ
,batch_normalization_13/AssignMovingAvg/sub_1Sub=batch_normalization_13/AssignMovingAvg/ReadVariableOp:value:04batch_normalization_13/FusedBatchNormV3:batch_mean:0*
T0*R
_classH
FDloc:@batch_normalization_13/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2.
,batch_normalization_13/AssignMovingAvg/sub_1║
*batch_normalization_13/AssignMovingAvg/mulMul0batch_normalization_13/AssignMovingAvg/sub_1:z:0.batch_normalization_13/AssignMovingAvg/sub:z:0*
T0*R
_classH
FDloc:@batch_normalization_13/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2,
*batch_normalization_13/AssignMovingAvg/mulУ
:batch_normalization_13/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp?batch_normalization_13_fusedbatchnormv3_readvariableop_resource.batch_normalization_13/AssignMovingAvg/mul:z:06^batch_normalization_13/AssignMovingAvg/ReadVariableOp7^batch_normalization_13/FusedBatchNormV3/ReadVariableOp*R
_classH
FDloc:@batch_normalization_13/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02<
:batch_normalization_13/AssignMovingAvg/AssignSubVariableOpч
.batch_normalization_13/AssignMovingAvg_1/sub/xConst*T
_classJ
HFloc:@batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  ђ?20
.batch_normalization_13/AssignMovingAvg_1/sub/x║
,batch_normalization_13/AssignMovingAvg_1/subSub7batch_normalization_13/AssignMovingAvg_1/sub/x:output:0%batch_normalization_13/Const:output:0*
T0*T
_classJ
HFloc:@batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2.
,batch_normalization_13/AssignMovingAvg_1/sub­
7batch_normalization_13/AssignMovingAvg_1/ReadVariableOpReadVariableOpAbatch_normalization_13_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype029
7batch_normalization_13/AssignMovingAvg_1/ReadVariableOpП
.batch_normalization_13/AssignMovingAvg_1/sub_1Sub?batch_normalization_13/AssignMovingAvg_1/ReadVariableOp:value:08batch_normalization_13/FusedBatchNormV3:batch_variance:0*
T0*T
_classJ
HFloc:@batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 20
.batch_normalization_13/AssignMovingAvg_1/sub_1─
,batch_normalization_13/AssignMovingAvg_1/mulMul2batch_normalization_13/AssignMovingAvg_1/sub_1:z:00batch_normalization_13/AssignMovingAvg_1/sub:z:0*
T0*T
_classJ
HFloc:@batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2.
,batch_normalization_13/AssignMovingAvg_1/mulШ
<batch_normalization_13/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpAbatch_normalization_13_fusedbatchnormv3_readvariableop_1_resource0batch_normalization_13/AssignMovingAvg_1/mul:z:08^batch_normalization_13/AssignMovingAvg_1/ReadVariableOp9^batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1*T
_classJ
HFloc:@batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02>
<batch_normalization_13/AssignMovingAvg_1/AssignSubVariableOpЌ
activation_17/ReluRelu+batch_normalization_13/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:          2
activation_17/ReluЮ
	add_6/addAddV2 activation_16/Relu:activations:0 activation_17/Relu:activations:0*
T0*/
_output_shapes
:          2
	add_6/add│
conv2d_15/Conv2D/ReadVariableOpReadVariableOp(conv2d_15_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02!
conv2d_15/Conv2D/ReadVariableOp╚
conv2d_15/Conv2DConv2Dadd_6/add:z:0'conv2d_15/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:          *
paddingSAME*
strides
2
conv2d_15/Conv2Dф
 conv2d_15/BiasAdd/ReadVariableOpReadVariableOp)conv2d_15_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_15/BiasAdd/ReadVariableOp░
conv2d_15/BiasAddBiasAddconv2d_15/Conv2D:output:0(conv2d_15/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          2
conv2d_15/BiasAdd╣
%batch_normalization_14/ReadVariableOpReadVariableOp.batch_normalization_14_readvariableop_resource*
_output_shapes
: *
dtype02'
%batch_normalization_14/ReadVariableOp┐
'batch_normalization_14/ReadVariableOp_1ReadVariableOp0batch_normalization_14_readvariableop_1_resource*
_output_shapes
: *
dtype02)
'batch_normalization_14/ReadVariableOp_1В
6batch_normalization_14/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_14_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype028
6batch_normalization_14/FusedBatchNormV3/ReadVariableOpЫ
8batch_normalization_14/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_14_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02:
8batch_normalization_14/FusedBatchNormV3/ReadVariableOp_1Н
'batch_normalization_14/FusedBatchNormV3FusedBatchNormV3conv2d_15/BiasAdd:output:0-batch_normalization_14/ReadVariableOp:value:0/batch_normalization_14/ReadVariableOp_1:value:0>batch_normalization_14/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_14/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:          : : : : :*
epsilon%oЃ:2)
'batch_normalization_14/FusedBatchNormV3Ђ
batch_normalization_14/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
batch_normalization_14/Constш
,batch_normalization_14/AssignMovingAvg/sub/xConst*R
_classH
FDloc:@batch_normalization_14/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  ђ?2.
,batch_normalization_14/AssignMovingAvg/sub/x▓
*batch_normalization_14/AssignMovingAvg/subSub5batch_normalization_14/AssignMovingAvg/sub/x:output:0%batch_normalization_14/Const:output:0*
T0*R
_classH
FDloc:@batch_normalization_14/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2,
*batch_normalization_14/AssignMovingAvg/subЖ
5batch_normalization_14/AssignMovingAvg/ReadVariableOpReadVariableOp?batch_normalization_14_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype027
5batch_normalization_14/AssignMovingAvg/ReadVariableOpЛ
,batch_normalization_14/AssignMovingAvg/sub_1Sub=batch_normalization_14/AssignMovingAvg/ReadVariableOp:value:04batch_normalization_14/FusedBatchNormV3:batch_mean:0*
T0*R
_classH
FDloc:@batch_normalization_14/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2.
,batch_normalization_14/AssignMovingAvg/sub_1║
*batch_normalization_14/AssignMovingAvg/mulMul0batch_normalization_14/AssignMovingAvg/sub_1:z:0.batch_normalization_14/AssignMovingAvg/sub:z:0*
T0*R
_classH
FDloc:@batch_normalization_14/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2,
*batch_normalization_14/AssignMovingAvg/mulУ
:batch_normalization_14/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp?batch_normalization_14_fusedbatchnormv3_readvariableop_resource.batch_normalization_14/AssignMovingAvg/mul:z:06^batch_normalization_14/AssignMovingAvg/ReadVariableOp7^batch_normalization_14/FusedBatchNormV3/ReadVariableOp*R
_classH
FDloc:@batch_normalization_14/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02<
:batch_normalization_14/AssignMovingAvg/AssignSubVariableOpч
.batch_normalization_14/AssignMovingAvg_1/sub/xConst*T
_classJ
HFloc:@batch_normalization_14/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  ђ?20
.batch_normalization_14/AssignMovingAvg_1/sub/x║
,batch_normalization_14/AssignMovingAvg_1/subSub7batch_normalization_14/AssignMovingAvg_1/sub/x:output:0%batch_normalization_14/Const:output:0*
T0*T
_classJ
HFloc:@batch_normalization_14/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2.
,batch_normalization_14/AssignMovingAvg_1/sub­
7batch_normalization_14/AssignMovingAvg_1/ReadVariableOpReadVariableOpAbatch_normalization_14_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype029
7batch_normalization_14/AssignMovingAvg_1/ReadVariableOpП
.batch_normalization_14/AssignMovingAvg_1/sub_1Sub?batch_normalization_14/AssignMovingAvg_1/ReadVariableOp:value:08batch_normalization_14/FusedBatchNormV3:batch_variance:0*
T0*T
_classJ
HFloc:@batch_normalization_14/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 20
.batch_normalization_14/AssignMovingAvg_1/sub_1─
,batch_normalization_14/AssignMovingAvg_1/mulMul2batch_normalization_14/AssignMovingAvg_1/sub_1:z:00batch_normalization_14/AssignMovingAvg_1/sub:z:0*
T0*T
_classJ
HFloc:@batch_normalization_14/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2.
,batch_normalization_14/AssignMovingAvg_1/mulШ
<batch_normalization_14/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpAbatch_normalization_14_fusedbatchnormv3_readvariableop_1_resource0batch_normalization_14/AssignMovingAvg_1/mul:z:08^batch_normalization_14/AssignMovingAvg_1/ReadVariableOp9^batch_normalization_14/FusedBatchNormV3/ReadVariableOp_1*T
_classJ
HFloc:@batch_normalization_14/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02>
<batch_normalization_14/AssignMovingAvg_1/AssignSubVariableOpЌ
activation_18/ReluRelu+batch_normalization_14/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:          2
activation_18/Reluі
	add_7/addAddV2add_6/add:z:0 activation_18/Relu:activations:0*
T0*/
_output_shapes
:          2
	add_7/addy
activation_19/ReluReluadd_7/add:z:0*
T0*/
_output_shapes
:          2
activation_19/Relu~
up_sampling2d_3/ShapeShape activation_19/Relu:activations:0*
T0*
_output_shapes
:2
up_sampling2d_3/Shapeћ
#up_sampling2d_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2%
#up_sampling2d_3/strided_slice/stackў
%up_sampling2d_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%up_sampling2d_3/strided_slice/stack_1ў
%up_sampling2d_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%up_sampling2d_3/strided_slice/stack_2«
up_sampling2d_3/strided_sliceStridedSliceup_sampling2d_3/Shape:output:0,up_sampling2d_3/strided_slice/stack:output:0.up_sampling2d_3/strided_slice/stack_1:output:0.up_sampling2d_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
up_sampling2d_3/strided_slice
up_sampling2d_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d_3/Constъ
up_sampling2d_3/mulMul&up_sampling2d_3/strided_slice:output:0up_sampling2d_3/Const:output:0*
T0*
_output_shapes
:2
up_sampling2d_3/mulё
,up_sampling2d_3/resize/ResizeNearestNeighborResizeNearestNeighbor activation_19/Relu:activations:0up_sampling2d_3/mul:z:0*
T0*/
_output_shapes
:          *
half_pixel_centers(2.
,up_sampling2d_3/resize/ResizeNearestNeighbor│
conv2d_16/Conv2D/ReadVariableOpReadVariableOp(conv2d_16_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02!
conv2d_16/Conv2D/ReadVariableOpЭ
conv2d_16/Conv2DConv2D=up_sampling2d_3/resize/ResizeNearestNeighbor:resized_images:0'conv2d_16/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:          *
paddingSAME*
strides
2
conv2d_16/Conv2Dф
 conv2d_16/BiasAdd/ReadVariableOpReadVariableOp)conv2d_16_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_16/BiasAdd/ReadVariableOp░
conv2d_16/BiasAddBiasAddconv2d_16/Conv2D:output:0(conv2d_16/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          2
conv2d_16/BiasAdd╣
%batch_normalization_15/ReadVariableOpReadVariableOp.batch_normalization_15_readvariableop_resource*
_output_shapes
: *
dtype02'
%batch_normalization_15/ReadVariableOp┐
'batch_normalization_15/ReadVariableOp_1ReadVariableOp0batch_normalization_15_readvariableop_1_resource*
_output_shapes
: *
dtype02)
'batch_normalization_15/ReadVariableOp_1В
6batch_normalization_15/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_15_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype028
6batch_normalization_15/FusedBatchNormV3/ReadVariableOpЫ
8batch_normalization_15/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_15_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02:
8batch_normalization_15/FusedBatchNormV3/ReadVariableOp_1Н
'batch_normalization_15/FusedBatchNormV3FusedBatchNormV3conv2d_16/BiasAdd:output:0-batch_normalization_15/ReadVariableOp:value:0/batch_normalization_15/ReadVariableOp_1:value:0>batch_normalization_15/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_15/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:          : : : : :*
epsilon%oЃ:2)
'batch_normalization_15/FusedBatchNormV3Ђ
batch_normalization_15/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
batch_normalization_15/Constш
,batch_normalization_15/AssignMovingAvg/sub/xConst*R
_classH
FDloc:@batch_normalization_15/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  ђ?2.
,batch_normalization_15/AssignMovingAvg/sub/x▓
*batch_normalization_15/AssignMovingAvg/subSub5batch_normalization_15/AssignMovingAvg/sub/x:output:0%batch_normalization_15/Const:output:0*
T0*R
_classH
FDloc:@batch_normalization_15/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2,
*batch_normalization_15/AssignMovingAvg/subЖ
5batch_normalization_15/AssignMovingAvg/ReadVariableOpReadVariableOp?batch_normalization_15_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype027
5batch_normalization_15/AssignMovingAvg/ReadVariableOpЛ
,batch_normalization_15/AssignMovingAvg/sub_1Sub=batch_normalization_15/AssignMovingAvg/ReadVariableOp:value:04batch_normalization_15/FusedBatchNormV3:batch_mean:0*
T0*R
_classH
FDloc:@batch_normalization_15/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2.
,batch_normalization_15/AssignMovingAvg/sub_1║
*batch_normalization_15/AssignMovingAvg/mulMul0batch_normalization_15/AssignMovingAvg/sub_1:z:0.batch_normalization_15/AssignMovingAvg/sub:z:0*
T0*R
_classH
FDloc:@batch_normalization_15/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2,
*batch_normalization_15/AssignMovingAvg/mulУ
:batch_normalization_15/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp?batch_normalization_15_fusedbatchnormv3_readvariableop_resource.batch_normalization_15/AssignMovingAvg/mul:z:06^batch_normalization_15/AssignMovingAvg/ReadVariableOp7^batch_normalization_15/FusedBatchNormV3/ReadVariableOp*R
_classH
FDloc:@batch_normalization_15/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02<
:batch_normalization_15/AssignMovingAvg/AssignSubVariableOpч
.batch_normalization_15/AssignMovingAvg_1/sub/xConst*T
_classJ
HFloc:@batch_normalization_15/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  ђ?20
.batch_normalization_15/AssignMovingAvg_1/sub/x║
,batch_normalization_15/AssignMovingAvg_1/subSub7batch_normalization_15/AssignMovingAvg_1/sub/x:output:0%batch_normalization_15/Const:output:0*
T0*T
_classJ
HFloc:@batch_normalization_15/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2.
,batch_normalization_15/AssignMovingAvg_1/sub­
7batch_normalization_15/AssignMovingAvg_1/ReadVariableOpReadVariableOpAbatch_normalization_15_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype029
7batch_normalization_15/AssignMovingAvg_1/ReadVariableOpП
.batch_normalization_15/AssignMovingAvg_1/sub_1Sub?batch_normalization_15/AssignMovingAvg_1/ReadVariableOp:value:08batch_normalization_15/FusedBatchNormV3:batch_variance:0*
T0*T
_classJ
HFloc:@batch_normalization_15/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 20
.batch_normalization_15/AssignMovingAvg_1/sub_1─
,batch_normalization_15/AssignMovingAvg_1/mulMul2batch_normalization_15/AssignMovingAvg_1/sub_1:z:00batch_normalization_15/AssignMovingAvg_1/sub:z:0*
T0*T
_classJ
HFloc:@batch_normalization_15/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2.
,batch_normalization_15/AssignMovingAvg_1/mulШ
<batch_normalization_15/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpAbatch_normalization_15_fusedbatchnormv3_readvariableop_1_resource0batch_normalization_15/AssignMovingAvg_1/mul:z:08^batch_normalization_15/AssignMovingAvg_1/ReadVariableOp9^batch_normalization_15/FusedBatchNormV3/ReadVariableOp_1*T
_classJ
HFloc:@batch_normalization_15/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02>
<batch_normalization_15/AssignMovingAvg_1/AssignSubVariableOpЌ
activation_20/ReluRelu+batch_normalization_15/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:          2
activation_20/Relu│
conv2d_17/Conv2D/ReadVariableOpReadVariableOp(conv2d_17_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_17/Conv2D/ReadVariableOp█
conv2d_17/Conv2DConv2D activation_20/Relu:activations:0'conv2d_17/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
paddingSAME*
strides
2
conv2d_17/Conv2Dф
 conv2d_17/BiasAdd/ReadVariableOpReadVariableOp)conv2d_17_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2d_17/BiasAdd/ReadVariableOp░
conv2d_17/BiasAddBiasAddconv2d_17/Conv2D:output:0(conv2d_17/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         2
conv2d_17/BiasAdd╣
%batch_normalization_16/ReadVariableOpReadVariableOp.batch_normalization_16_readvariableop_resource*
_output_shapes
:*
dtype02'
%batch_normalization_16/ReadVariableOp┐
'batch_normalization_16/ReadVariableOp_1ReadVariableOp0batch_normalization_16_readvariableop_1_resource*
_output_shapes
:*
dtype02)
'batch_normalization_16/ReadVariableOp_1В
6batch_normalization_16/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_16_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype028
6batch_normalization_16/FusedBatchNormV3/ReadVariableOpЫ
8batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_16_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02:
8batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1Н
'batch_normalization_16/FusedBatchNormV3FusedBatchNormV3conv2d_17/BiasAdd:output:0-batch_normalization_16/ReadVariableOp:value:0/batch_normalization_16/ReadVariableOp_1:value:0>batch_normalization_16/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         :::::*
epsilon%oЃ:2)
'batch_normalization_16/FusedBatchNormV3Ђ
batch_normalization_16/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
batch_normalization_16/Constш
,batch_normalization_16/AssignMovingAvg/sub/xConst*R
_classH
FDloc:@batch_normalization_16/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  ђ?2.
,batch_normalization_16/AssignMovingAvg/sub/x▓
*batch_normalization_16/AssignMovingAvg/subSub5batch_normalization_16/AssignMovingAvg/sub/x:output:0%batch_normalization_16/Const:output:0*
T0*R
_classH
FDloc:@batch_normalization_16/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2,
*batch_normalization_16/AssignMovingAvg/subЖ
5batch_normalization_16/AssignMovingAvg/ReadVariableOpReadVariableOp?batch_normalization_16_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype027
5batch_normalization_16/AssignMovingAvg/ReadVariableOpЛ
,batch_normalization_16/AssignMovingAvg/sub_1Sub=batch_normalization_16/AssignMovingAvg/ReadVariableOp:value:04batch_normalization_16/FusedBatchNormV3:batch_mean:0*
T0*R
_classH
FDloc:@batch_normalization_16/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:2.
,batch_normalization_16/AssignMovingAvg/sub_1║
*batch_normalization_16/AssignMovingAvg/mulMul0batch_normalization_16/AssignMovingAvg/sub_1:z:0.batch_normalization_16/AssignMovingAvg/sub:z:0*
T0*R
_classH
FDloc:@batch_normalization_16/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:2,
*batch_normalization_16/AssignMovingAvg/mulУ
:batch_normalization_16/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp?batch_normalization_16_fusedbatchnormv3_readvariableop_resource.batch_normalization_16/AssignMovingAvg/mul:z:06^batch_normalization_16/AssignMovingAvg/ReadVariableOp7^batch_normalization_16/FusedBatchNormV3/ReadVariableOp*R
_classH
FDloc:@batch_normalization_16/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02<
:batch_normalization_16/AssignMovingAvg/AssignSubVariableOpч
.batch_normalization_16/AssignMovingAvg_1/sub/xConst*T
_classJ
HFloc:@batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  ђ?20
.batch_normalization_16/AssignMovingAvg_1/sub/x║
,batch_normalization_16/AssignMovingAvg_1/subSub7batch_normalization_16/AssignMovingAvg_1/sub/x:output:0%batch_normalization_16/Const:output:0*
T0*T
_classJ
HFloc:@batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2.
,batch_normalization_16/AssignMovingAvg_1/sub­
7batch_normalization_16/AssignMovingAvg_1/ReadVariableOpReadVariableOpAbatch_normalization_16_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype029
7batch_normalization_16/AssignMovingAvg_1/ReadVariableOpП
.batch_normalization_16/AssignMovingAvg_1/sub_1Sub?batch_normalization_16/AssignMovingAvg_1/ReadVariableOp:value:08batch_normalization_16/FusedBatchNormV3:batch_variance:0*
T0*T
_classJ
HFloc:@batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:20
.batch_normalization_16/AssignMovingAvg_1/sub_1─
,batch_normalization_16/AssignMovingAvg_1/mulMul2batch_normalization_16/AssignMovingAvg_1/sub_1:z:00batch_normalization_16/AssignMovingAvg_1/sub:z:0*
T0*T
_classJ
HFloc:@batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:2.
,batch_normalization_16/AssignMovingAvg_1/mulШ
<batch_normalization_16/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpAbatch_normalization_16_fusedbatchnormv3_readvariableop_1_resource0batch_normalization_16/AssignMovingAvg_1/mul:z:08^batch_normalization_16/AssignMovingAvg_1/ReadVariableOp9^batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1*T
_classJ
HFloc:@batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02>
<batch_normalization_16/AssignMovingAvg_1/AssignSubVariableOpЌ
activation_21/ReluRelu+batch_normalization_16/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:         2
activation_21/Relu┌
IdentityIdentity activation_21/Relu:activations:0;^batch_normalization_10/AssignMovingAvg/AssignSubVariableOp=^batch_normalization_10/AssignMovingAvg_1/AssignSubVariableOp;^batch_normalization_11/AssignMovingAvg/AssignSubVariableOp=^batch_normalization_11/AssignMovingAvg_1/AssignSubVariableOp;^batch_normalization_12/AssignMovingAvg/AssignSubVariableOp=^batch_normalization_12/AssignMovingAvg_1/AssignSubVariableOp;^batch_normalization_13/AssignMovingAvg/AssignSubVariableOp=^batch_normalization_13/AssignMovingAvg_1/AssignSubVariableOp;^batch_normalization_14/AssignMovingAvg/AssignSubVariableOp=^batch_normalization_14/AssignMovingAvg_1/AssignSubVariableOp;^batch_normalization_15/AssignMovingAvg/AssignSubVariableOp=^batch_normalization_15/AssignMovingAvg_1/AssignSubVariableOp;^batch_normalization_16/AssignMovingAvg/AssignSubVariableOp=^batch_normalization_16/AssignMovingAvg_1/AssignSubVariableOp:^batch_normalization_9/AssignMovingAvg/AssignSubVariableOp<^batch_normalization_9/AssignMovingAvg_1/AssignSubVariableOp*
T0*/
_output_shapes
:         2

Identity"
identityIdentity:output:0*­
_input_shapesя
█:         ::::::::::::::::::::::::::::::::::::::::::::::::2x
:batch_normalization_10/AssignMovingAvg/AssignSubVariableOp:batch_normalization_10/AssignMovingAvg/AssignSubVariableOp2|
<batch_normalization_10/AssignMovingAvg_1/AssignSubVariableOp<batch_normalization_10/AssignMovingAvg_1/AssignSubVariableOp2x
:batch_normalization_11/AssignMovingAvg/AssignSubVariableOp:batch_normalization_11/AssignMovingAvg/AssignSubVariableOp2|
<batch_normalization_11/AssignMovingAvg_1/AssignSubVariableOp<batch_normalization_11/AssignMovingAvg_1/AssignSubVariableOp2x
:batch_normalization_12/AssignMovingAvg/AssignSubVariableOp:batch_normalization_12/AssignMovingAvg/AssignSubVariableOp2|
<batch_normalization_12/AssignMovingAvg_1/AssignSubVariableOp<batch_normalization_12/AssignMovingAvg_1/AssignSubVariableOp2x
:batch_normalization_13/AssignMovingAvg/AssignSubVariableOp:batch_normalization_13/AssignMovingAvg/AssignSubVariableOp2|
<batch_normalization_13/AssignMovingAvg_1/AssignSubVariableOp<batch_normalization_13/AssignMovingAvg_1/AssignSubVariableOp2x
:batch_normalization_14/AssignMovingAvg/AssignSubVariableOp:batch_normalization_14/AssignMovingAvg/AssignSubVariableOp2|
<batch_normalization_14/AssignMovingAvg_1/AssignSubVariableOp<batch_normalization_14/AssignMovingAvg_1/AssignSubVariableOp2x
:batch_normalization_15/AssignMovingAvg/AssignSubVariableOp:batch_normalization_15/AssignMovingAvg/AssignSubVariableOp2|
<batch_normalization_15/AssignMovingAvg_1/AssignSubVariableOp<batch_normalization_15/AssignMovingAvg_1/AssignSubVariableOp2x
:batch_normalization_16/AssignMovingAvg/AssignSubVariableOp:batch_normalization_16/AssignMovingAvg/AssignSubVariableOp2|
<batch_normalization_16/AssignMovingAvg_1/AssignSubVariableOp<batch_normalization_16/AssignMovingAvg_1/AssignSubVariableOp2v
9batch_normalization_9/AssignMovingAvg/AssignSubVariableOp9batch_normalization_9/AssignMovingAvg/AssignSubVariableOp2z
;batch_normalization_9/AssignMovingAvg_1/AssignSubVariableOp;batch_normalization_9/AssignMovingAvg_1/AssignSubVariableOp:W S
/
_output_shapes
:         
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
: :!

_output_shapes
: :"

_output_shapes
: :#

_output_shapes
: :$

_output_shapes
: :%

_output_shapes
: :&

_output_shapes
: :'

_output_shapes
: :(

_output_shapes
: :)

_output_shapes
: :*

_output_shapes
: :+

_output_shapes
: :,

_output_shapes
: :-

_output_shapes
: :.

_output_shapes
: :/

_output_shapes
: :0

_output_shapes
: 
╔
І
R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_254569

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityѕt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOpГ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1╩
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:          : : : : :*
epsilon%oЃ:*
is_training( 2
FusedBatchNormV3p
IdentityIdentityFusedBatchNormV3:y:0*
T0*/
_output_shapes
:          2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:          :::::W S
/
_output_shapes
:          
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
њ
І
R__inference_batch_normalization_11_layer_call_and_return_conditional_losses_253626

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityѕt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOpГ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1▄
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                            : : : : :*
epsilon%oЃ:*
is_training( 2
FusedBatchNormV3ѓ
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+                            2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                            :::::i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
├
m
A__inference_add_7_layer_call_and_return_conditional_losses_257425
inputs_0
inputs_1
identitys
addAddV2inputs_0inputs_1*
T0*A
_output_shapes/
-:+                            2
addu
IdentityIdentityadd:z:0*
T0*A
_output_shapes/
-:+                            2

Identity"
identityIdentity:output:0*m
_input_shapes\
Z:+                            :+                            :k g
A
_output_shapes/
-:+                            
"
_user_specified_name
inputs/0:kg
A
_output_shapes/
-:+                            
"
_user_specified_name
inputs/1
ж

*__inference_conv2d_15_layer_call_fn_253971

inputs
unknown
	unknown_0
identityѕбStatefulPartitionedCallШ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*A
_output_shapes/
-:+                            *$
_read_only_resource_inputs
*3
config_proto#!

CPU

GPU2*0,1,2,3J 8*N
fIRG
E__inference_conv2d_15_layer_call_and_return_conditional_losses_2539612
StatefulPartitionedCallе
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                            2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+                            ::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
њ
І
R__inference_batch_normalization_13_layer_call_and_return_conditional_losses_257274

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityѕt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOpГ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1▄
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                            : : : : :*
epsilon%oЃ:*
is_training( 2
FusedBatchNormV3ѓ
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+                            2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                            :::::i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
У$
┘
R__inference_batch_normalization_16_layer_call_and_return_conditional_losses_257581

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityѕб#AssignMovingAvg/AssignSubVariableOpб%AssignMovingAvg_1/AssignSubVariableOpt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOpГ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1╔
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           :::::*
epsilon%oЃ:2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
Const░
AssignMovingAvg/sub/xConst*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  ђ?2
AssignMovingAvg/sub/x┐
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const:output:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/subЦ
AssignMovingAvg/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOpя
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:2
AssignMovingAvg/sub_1К
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:2
AssignMovingAvg/mulК
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp(fusedbatchnormv3_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpХ
AssignMovingAvg_1/sub/xConst*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  ђ?2
AssignMovingAvg_1/sub/xК
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const:output:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/subФ
 AssignMovingAvg_1/ReadVariableOpReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOpЖ
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:2
AssignMovingAvg_1/sub_1Л
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:2
AssignMovingAvg_1/mulН
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp*fusedbatchnormv3_readvariableop_1_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpл
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                           ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
У$
┘
R__inference_batch_normalization_13_layer_call_and_return_conditional_losses_253908

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityѕб#AssignMovingAvg/AssignSubVariableOpб%AssignMovingAvg_1/AssignSubVariableOpt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOpГ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1╔
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                            : : : : :*
epsilon%oЃ:2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
Const░
AssignMovingAvg/sub/xConst*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  ђ?2
AssignMovingAvg/sub/x┐
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const:output:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/subЦ
AssignMovingAvg/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOpя
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/sub_1К
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/mulК
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp(fusedbatchnormv3_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpХ
AssignMovingAvg_1/sub/xConst*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  ђ?2
AssignMovingAvg_1/sub/xК
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const:output:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/subФ
 AssignMovingAvg_1/ReadVariableOpReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOpЖ
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/sub_1Л
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/mulН
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp*fusedbatchnormv3_readvariableop_1_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpл
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*A
_output_shapes/
-:+                            2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                            ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
╗
k
A__inference_add_6_layer_call_and_return_conditional_losses_254876

inputs
inputs_1
identityq
addAddV2inputsinputs_1*
T0*A
_output_shapes/
-:+                            2
addu
IdentityIdentityadd:z:0*
T0*A
_output_shapes/
-:+                            2

Identity"
identityIdentity:output:0*m
_input_shapes\
Z:+                            :+                            :i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs:ie
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
│
н
(__inference_model_2_layer_call_fn_256566

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34

unknown_35

unknown_36

unknown_37

unknown_38

unknown_39

unknown_40

unknown_41

unknown_42

unknown_43

unknown_44

unknown_45

unknown_46
identityѕбStatefulPartitionedCall№
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46*<
Tin5
321*
Tout
2*A
_output_shapes/
-:+                           *R
_read_only_resource_inputs4
20	
 !"#$%&'()*+,-./0*3
config_proto#!

CPU

GPU2*0,1,2,3J 8*L
fGRE
C__inference_model_2_layer_call_and_return_conditional_losses_2555742
StatefulPartitionedCallе
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*­
_input_shapesя
█:         ::::::::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
: :!

_output_shapes
: :"

_output_shapes
: :#

_output_shapes
: :$

_output_shapes
: :%

_output_shapes
: :&

_output_shapes
: :'

_output_shapes
: :(

_output_shapes
: :)

_output_shapes
: :*

_output_shapes
: :+

_output_shapes
: :,

_output_shapes
: :-

_output_shapes
: :.

_output_shapes
: :/

_output_shapes
: :0

_output_shapes
: 
њ
І
R__inference_batch_normalization_14_layer_call_and_return_conditional_losses_254086

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityѕt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOpГ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1▄
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                            : : : : :*
epsilon%oЃ:*
is_training( 2
FusedBatchNormV3ѓ
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+                            2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                            :::::i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
■
Е
6__inference_batch_normalization_9_layer_call_fn_256653

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityѕбStatefulPartitionedCallю
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*A
_output_shapes/
-:+                            *&
_read_only_resource_inputs
*3
config_proto#!

CPU

GPU2*0,1,2,3J 8*Z
fURS
Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_2533322
StatefulPartitionedCallе
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                            2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                            ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
О
e
I__inference_activation_12_layer_call_and_return_conditional_losses_256733

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:          2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:          2

Identity"
identityIdentity:output:0*.
_input_shapes
:          :W S
/
_output_shapes
:          
 
_user_specified_nameinputs
■
ф
7__inference_batch_normalization_16_layer_call_fn_257612

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityѕбStatefulPartitionedCallЏ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*A
_output_shapes/
-:+                           *$
_read_only_resource_inputs
*3
config_proto#!

CPU

GPU2*0,1,2,3J 8*[
fVRT
R__inference_batch_normalization_16_layer_call_and_return_conditional_losses_2543682
StatefulPartitionedCallе
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                           ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
ж

*__inference_conv2d_13_layer_call_fn_253677

inputs
unknown
	unknown_0
identityѕбStatefulPartitionedCallШ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*A
_output_shapes/
-:+                            *$
_read_only_resource_inputs
*3
config_proto#!

CPU

GPU2*0,1,2,3J 8*N
fIRG
E__inference_conv2d_13_layer_call_and_return_conditional_losses_2536672
StatefulPartitionedCallе
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                            2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+                            ::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
У$
┘
R__inference_batch_normalization_15_layer_call_and_return_conditional_losses_254221

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityѕб#AssignMovingAvg/AssignSubVariableOpб%AssignMovingAvg_1/AssignSubVariableOpt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOpГ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1╔
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                            : : : : :*
epsilon%oЃ:2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
Const░
AssignMovingAvg/sub/xConst*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  ђ?2
AssignMovingAvg/sub/x┐
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const:output:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/subЦ
AssignMovingAvg/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOpя
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/sub_1К
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/mulК
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp(fusedbatchnormv3_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpХ
AssignMovingAvg_1/sub/xConst*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  ђ?2
AssignMovingAvg_1/sub/xК
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const:output:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/subФ
 AssignMovingAvg_1/ReadVariableOpReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOpЖ
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/sub_1Л
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/mulН
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp*fusedbatchnormv3_readvariableop_1_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpл
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*A
_output_shapes/
-:+                            2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                            ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
њ
І
R__inference_batch_normalization_12_layer_call_and_return_conditional_losses_257177

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityѕt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOpГ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1▄
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                            : : : : :*
epsilon%oЃ:*
is_training( 2
FusedBatchNormV3ѓ
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+                            2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                            :::::i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
ђ	
ф
7__inference_batch_normalization_12_layer_call_fn_257203

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityѕбStatefulPartitionedCallЮ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*A
_output_shapes/
-:+                            *&
_read_only_resource_inputs
*3
config_proto#!

CPU

GPU2*0,1,2,3J 8*[
fVRT
R__inference_batch_normalization_12_layer_call_and_return_conditional_losses_2537922
StatefulPartitionedCallе
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                            2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                            ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
а
e
I__inference_activation_16_layer_call_and_return_conditional_losses_254809

inputs
identityh
ReluReluinputs*
T0*A
_output_shapes/
-:+                            2
Reluђ
IdentityIdentityRelu:activations:0*
T0*A
_output_shapes/
-:+                            2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+                            :i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
ђ	
ф
7__inference_batch_normalization_16_layer_call_fn_257625

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityѕбStatefulPartitionedCallЮ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*A
_output_shapes/
-:+                           *&
_read_only_resource_inputs
*3
config_proto#!

CPU

GPU2*0,1,2,3J 8*[
fVRT
R__inference_batch_normalization_16_layer_call_and_return_conditional_losses_2543992
StatefulPartitionedCallе
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                           ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
њ
І
R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_256874

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityѕt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOpГ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1▄
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                            : : : : :*
epsilon%oЃ:*
is_training( 2
FusedBatchNormV3ѓ
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+                            2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                            :::::i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
њ
І
R__inference_batch_normalization_16_layer_call_and_return_conditional_losses_254399

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityѕt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOpГ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1▄
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           :::::*
epsilon%oЃ:*
is_training( 2
FusedBatchNormV3ѓ
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                           :::::i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
ђ	
ф
7__inference_batch_normalization_13_layer_call_fn_257300

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityѕбStatefulPartitionedCallЮ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*A
_output_shapes/
-:+                            *&
_read_only_resource_inputs
*3
config_proto#!

CPU

GPU2*0,1,2,3J 8*[
fVRT
R__inference_batch_normalization_13_layer_call_and_return_conditional_losses_2539392
StatefulPartitionedCallе
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                            2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                            ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
а$
┘
R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_254551

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityѕб#AssignMovingAvg/AssignSubVariableOpб%AssignMovingAvg_1/AssignSubVariableOpt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOpГ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1и
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:          : : : : :*
epsilon%oЃ:2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
Const░
AssignMovingAvg/sub/xConst*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  ђ?2
AssignMovingAvg/sub/x┐
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const:output:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/subЦ
AssignMovingAvg/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOpя
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/sub_1К
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/mulК
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp(fusedbatchnormv3_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpХ
AssignMovingAvg_1/sub/xConst*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  ђ?2
AssignMovingAvg_1/sub/xК
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const:output:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/subФ
 AssignMovingAvg_1/ReadVariableOpReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOpЖ
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/sub_1Л
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/mulН
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp*fusedbatchnormv3_readvariableop_1_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpЙ
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*/
_output_shapes
:          2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:          ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:W S
/
_output_shapes
:          
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
њ
І
R__inference_batch_normalization_16_layer_call_and_return_conditional_losses_257599

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityѕt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOpГ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1▄
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           :::::*
epsilon%oЃ:*
is_training( 2
FusedBatchNormV3ѓ
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                           :::::i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
И
ф
7__inference_batch_normalization_11_layer_call_fn_257009

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityѕбStatefulPartitionedCallІ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*/
_output_shapes
:          *&
_read_only_resource_inputs
*3
config_proto#!

CPU

GPU2*0,1,2,3J 8*[
fVRT
R__inference_batch_normalization_11_layer_call_and_return_conditional_losses_2546862
StatefulPartitionedCallќ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:          2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:          ::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:          
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
а$
┘
R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_256781

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityѕб#AssignMovingAvg/AssignSubVariableOpб%AssignMovingAvg_1/AssignSubVariableOpt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOpГ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1и
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:          : : : : :*
epsilon%oЃ:2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
Const░
AssignMovingAvg/sub/xConst*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  ђ?2
AssignMovingAvg/sub/x┐
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const:output:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/subЦ
AssignMovingAvg/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOpя
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/sub_1К
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/mulК
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp(fusedbatchnormv3_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpХ
AssignMovingAvg_1/sub/xConst*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  ђ?2
AssignMovingAvg_1/sub/xК
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const:output:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/subФ
 AssignMovingAvg_1/ReadVariableOpReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOpЖ
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/sub_1Л
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/mulН
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp*fusedbatchnormv3_readvariableop_1_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpЙ
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*/
_output_shapes
:          2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:          ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:W S
/
_output_shapes
:          
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
а
e
I__inference_activation_16_layer_call_and_return_conditional_losses_257208

inputs
identityh
ReluReluinputs*
T0*A
_output_shapes/
-:+                            2
Reluђ
IdentityIdentityRelu:activations:0*
T0*A
_output_shapes/
-:+                            2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+                            :i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
а
e
I__inference_activation_21_layer_call_and_return_conditional_losses_255065

inputs
identityh
ReluReluinputs*
T0*A
_output_shapes/
-:+                           2
Reluђ
IdentityIdentityRelu:activations:0*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+                           :i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
╚
і
Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_256702

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityѕt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOpГ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1╩
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:          : : : : :*
epsilon%oЃ:*
is_training( 2
FusedBatchNormV3p
IdentityIdentityFusedBatchNormV3:y:0*
T0*/
_output_shapes
:          2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:          :::::W S
/
_output_shapes
:          
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
У$
┘
R__inference_batch_normalization_14_layer_call_and_return_conditional_losses_254055

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityѕб#AssignMovingAvg/AssignSubVariableOpб%AssignMovingAvg_1/AssignSubVariableOpt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOpГ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1╔
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                            : : : : :*
epsilon%oЃ:2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
Const░
AssignMovingAvg/sub/xConst*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  ђ?2
AssignMovingAvg/sub/x┐
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const:output:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/subЦ
AssignMovingAvg/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOpя
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/sub_1К
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/mulК
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp(fusedbatchnormv3_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpХ
AssignMovingAvg_1/sub/xConst*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  ђ?2
AssignMovingAvg_1/sub/xК
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const:output:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/subФ
 AssignMovingAvg_1/ReadVariableOpReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOpЖ
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/sub_1Л
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/mulН
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp*fusedbatchnormv3_readvariableop_1_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpл
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*A
_output_shapes/
-:+                            2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                            ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
У$
┘
R__inference_batch_normalization_14_layer_call_and_return_conditional_losses_257365

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityѕб#AssignMovingAvg/AssignSubVariableOpб%AssignMovingAvg_1/AssignSubVariableOpt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOpГ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1╔
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                            : : : : :*
epsilon%oЃ:2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
Const░
AssignMovingAvg/sub/xConst*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  ђ?2
AssignMovingAvg/sub/x┐
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const:output:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/subЦ
AssignMovingAvg/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOpя
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/sub_1К
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/mulК
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp(fusedbatchnormv3_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpХ
AssignMovingAvg_1/sub/xConst*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  ђ?2
AssignMovingAvg_1/sub/xК
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const:output:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/subФ
 AssignMovingAvg_1/ReadVariableOpReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOpЖ
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/sub_1Л
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/mulН
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp*fusedbatchnormv3_readvariableop_1_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpл
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*A
_output_shapes/
-:+                            2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                            ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
╝	
Г
E__inference_conv2d_11_layer_call_and_return_conditional_losses_253354

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityѕЋ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02
Conv2D/ReadVariableOpх
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                            *
paddingSAME*
strides
2
Conv2Dї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOpџ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                            2	
BiasAdd~
IdentityIdentityBiasAdd:output:0*
T0*A
_output_shapes/
-:+                            2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+                            :::i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
Б
н
(__inference_model_2_layer_call_fn_256465

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34

unknown_35

unknown_36

unknown_37

unknown_38

unknown_39

unknown_40

unknown_41

unknown_42

unknown_43

unknown_44

unknown_45

unknown_46
identityѕбStatefulPartitionedCall▀
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46*<
Tin5
321*
Tout
2*A
_output_shapes/
-:+                           *B
_read_only_resource_inputs$
" 	
 !"%&'(+,-.*3
config_proto#!

CPU

GPU2*0,1,2,3J 8*L
fGRE
C__inference_model_2_layer_call_and_return_conditional_losses_2553412
StatefulPartitionedCallе
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*­
_input_shapesя
█:         ::::::::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
: :!

_output_shapes
: :"

_output_shapes
: :#

_output_shapes
: :$

_output_shapes
: :%

_output_shapes
: :&

_output_shapes
: :'

_output_shapes
: :(

_output_shapes
: :)

_output_shapes
: :*

_output_shapes
: :+

_output_shapes
: :,

_output_shapes
: :-

_output_shapes
: :.

_output_shapes
: :/

_output_shapes
: :0

_output_shapes
: 
ж

*__inference_conv2d_11_layer_call_fn_253364

inputs
unknown
	unknown_0
identityѕбStatefulPartitionedCallШ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*A
_output_shapes/
-:+                            *$
_read_only_resource_inputs
*3
config_proto#!

CPU

GPU2*0,1,2,3J 8*N
fIRG
E__inference_conv2d_11_layer_call_and_return_conditional_losses_2533542
StatefulPartitionedCallе
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                            2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+                            ::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
ћ
L
0__inference_up_sampling2d_2_layer_call_fn_253656

inputs
identityМ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*J
_output_shapes8
6:4                                    * 
_read_only_resource_inputs
 *3
config_proto#!

CPU

GPU2*0,1,2,3J 8*T
fORM
K__inference_up_sampling2d_2_layer_call_and_return_conditional_losses_2536502
PartitionedCallЈ
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4                                    2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
Ша
к
C__inference_model_2_layer_call_and_return_conditional_losses_255074
input_3
conv2d_10_254414
conv2d_10_254416 
batch_normalization_9_254494 
batch_normalization_9_254496 
batch_normalization_9_254498 
batch_normalization_9_254500
conv2d_11_254516
conv2d_11_254518!
batch_normalization_10_254596!
batch_normalization_10_254598!
batch_normalization_10_254600!
batch_normalization_10_254602
conv2d_12_254633
conv2d_12_254635!
batch_normalization_11_254713!
batch_normalization_11_254715!
batch_normalization_11_254717!
batch_normalization_11_254719
conv2d_13_254764
conv2d_13_254766!
batch_normalization_12_254795!
batch_normalization_12_254797!
batch_normalization_12_254799!
batch_normalization_12_254801
conv2d_14_254817
conv2d_14_254819!
batch_normalization_13_254848!
batch_normalization_13_254850!
batch_normalization_13_254852!
batch_normalization_13_254854
conv2d_15_254885
conv2d_15_254887!
batch_normalization_14_254916!
batch_normalization_14_254918!
batch_normalization_14_254920!
batch_normalization_14_254922
conv2d_16_254967
conv2d_16_254969!
batch_normalization_15_254998!
batch_normalization_15_255000!
batch_normalization_15_255002!
batch_normalization_15_255004
conv2d_17_255020
conv2d_17_255022!
batch_normalization_16_255051!
batch_normalization_16_255053!
batch_normalization_16_255055!
batch_normalization_16_255057
identityѕб.batch_normalization_10/StatefulPartitionedCallб.batch_normalization_11/StatefulPartitionedCallб.batch_normalization_12/StatefulPartitionedCallб.batch_normalization_13/StatefulPartitionedCallб.batch_normalization_14/StatefulPartitionedCallб.batch_normalization_15/StatefulPartitionedCallб.batch_normalization_16/StatefulPartitionedCallб-batch_normalization_9/StatefulPartitionedCallб!conv2d_10/StatefulPartitionedCallб!conv2d_11/StatefulPartitionedCallб!conv2d_12/StatefulPartitionedCallб!conv2d_13/StatefulPartitionedCallб!conv2d_14/StatefulPartitionedCallб!conv2d_15/StatefulPartitionedCallб!conv2d_16/StatefulPartitionedCallб!conv2d_17/StatefulPartitionedCallЅ
!conv2d_10/StatefulPartitionedCallStatefulPartitionedCallinput_3conv2d_10_254414conv2d_10_254416*
Tin
2*
Tout
2*/
_output_shapes
:          *$
_read_only_resource_inputs
*3
config_proto#!

CPU

GPU2*0,1,2,3J 8*N
fIRG
E__inference_conv2d_10_layer_call_and_return_conditional_losses_2532072#
!conv2d_10/StatefulPartitionedCallд
-batch_normalization_9/StatefulPartitionedCallStatefulPartitionedCall*conv2d_10/StatefulPartitionedCall:output:0batch_normalization_9_254494batch_normalization_9_254496batch_normalization_9_254498batch_normalization_9_254500*
Tin	
2*
Tout
2*/
_output_shapes
:          *$
_read_only_resource_inputs
*3
config_proto#!

CPU

GPU2*0,1,2,3J 8*Z
fURS
Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_2544492/
-batch_normalization_9/StatefulPartitionedCallѓ
activation_12/PartitionedCallPartitionedCall6batch_normalization_9/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:          * 
_read_only_resource_inputs
 *3
config_proto#!

CPU

GPU2*0,1,2,3J 8*R
fMRK
I__inference_activation_12_layer_call_and_return_conditional_losses_2545082
activation_12/PartitionedCallе
!conv2d_11/StatefulPartitionedCallStatefulPartitionedCall&activation_12/PartitionedCall:output:0conv2d_11_254516conv2d_11_254518*
Tin
2*
Tout
2*/
_output_shapes
:          *$
_read_only_resource_inputs
*3
config_proto#!

CPU

GPU2*0,1,2,3J 8*N
fIRG
E__inference_conv2d_11_layer_call_and_return_conditional_losses_2533542#
!conv2d_11/StatefulPartitionedCallГ
.batch_normalization_10/StatefulPartitionedCallStatefulPartitionedCall*conv2d_11/StatefulPartitionedCall:output:0batch_normalization_10_254596batch_normalization_10_254598batch_normalization_10_254600batch_normalization_10_254602*
Tin	
2*
Tout
2*/
_output_shapes
:          *$
_read_only_resource_inputs
*3
config_proto#!

CPU

GPU2*0,1,2,3J 8*[
fVRT
R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_25455120
.batch_normalization_10/StatefulPartitionedCallЃ
activation_13/PartitionedCallPartitionedCall7batch_normalization_10/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:          * 
_read_only_resource_inputs
 *3
config_proto#!

CPU

GPU2*0,1,2,3J 8*R
fMRK
I__inference_activation_13_layer_call_and_return_conditional_losses_2546102
activation_13/PartitionedCallЃ
add_4/PartitionedCallPartitionedCall&activation_12/PartitionedCall:output:0&activation_13/PartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:          * 
_read_only_resource_inputs
 *3
config_proto#!

CPU

GPU2*0,1,2,3J 8*J
fERC
A__inference_add_4_layer_call_and_return_conditional_losses_2546242
add_4/PartitionedCallа
!conv2d_12/StatefulPartitionedCallStatefulPartitionedCalladd_4/PartitionedCall:output:0conv2d_12_254633conv2d_12_254635*
Tin
2*
Tout
2*/
_output_shapes
:          *$
_read_only_resource_inputs
*3
config_proto#!

CPU

GPU2*0,1,2,3J 8*N
fIRG
E__inference_conv2d_12_layer_call_and_return_conditional_losses_2535012#
!conv2d_12/StatefulPartitionedCallГ
.batch_normalization_11/StatefulPartitionedCallStatefulPartitionedCall*conv2d_12/StatefulPartitionedCall:output:0batch_normalization_11_254713batch_normalization_11_254715batch_normalization_11_254717batch_normalization_11_254719*
Tin	
2*
Tout
2*/
_output_shapes
:          *$
_read_only_resource_inputs
*3
config_proto#!

CPU

GPU2*0,1,2,3J 8*[
fVRT
R__inference_batch_normalization_11_layer_call_and_return_conditional_losses_25466820
.batch_normalization_11/StatefulPartitionedCallЃ
activation_14/PartitionedCallPartitionedCall7batch_normalization_11/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:          * 
_read_only_resource_inputs
 *3
config_proto#!

CPU

GPU2*0,1,2,3J 8*R
fMRK
I__inference_activation_14_layer_call_and_return_conditional_losses_2547272
activation_14/PartitionedCallч
add_5/PartitionedCallPartitionedCalladd_4/PartitionedCall:output:0&activation_14/PartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:          * 
_read_only_resource_inputs
 *3
config_proto#!

CPU

GPU2*0,1,2,3J 8*J
fERC
A__inference_add_5_layer_call_and_return_conditional_losses_2547412
add_5/PartitionedCallЖ
activation_15/PartitionedCallPartitionedCalladd_5/PartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:          * 
_read_only_resource_inputs
 *3
config_proto#!

CPU

GPU2*0,1,2,3J 8*R
fMRK
I__inference_activation_15_layer_call_and_return_conditional_losses_2547552
activation_15/PartitionedCallі
up_sampling2d_2/PartitionedCallPartitionedCall&activation_15/PartitionedCall:output:0*
Tin
2*
Tout
2*A
_output_shapes/
-:+                            * 
_read_only_resource_inputs
 *3
config_proto#!

CPU

GPU2*0,1,2,3J 8*T
fORM
K__inference_up_sampling2d_2_layer_call_and_return_conditional_losses_2536502!
up_sampling2d_2/PartitionedCall╝
!conv2d_13/StatefulPartitionedCallStatefulPartitionedCall(up_sampling2d_2/PartitionedCall:output:0conv2d_13_254764conv2d_13_254766*
Tin
2*
Tout
2*A
_output_shapes/
-:+                            *$
_read_only_resource_inputs
*3
config_proto#!

CPU

GPU2*0,1,2,3J 8*N
fIRG
E__inference_conv2d_13_layer_call_and_return_conditional_losses_2536672#
!conv2d_13/StatefulPartitionedCall┐
.batch_normalization_12/StatefulPartitionedCallStatefulPartitionedCall*conv2d_13/StatefulPartitionedCall:output:0batch_normalization_12_254795batch_normalization_12_254797batch_normalization_12_254799batch_normalization_12_254801*
Tin	
2*
Tout
2*A
_output_shapes/
-:+                            *$
_read_only_resource_inputs
*3
config_proto#!

CPU

GPU2*0,1,2,3J 8*[
fVRT
R__inference_batch_normalization_12_layer_call_and_return_conditional_losses_25376120
.batch_normalization_12/StatefulPartitionedCallЋ
activation_16/PartitionedCallPartitionedCall7batch_normalization_12/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*A
_output_shapes/
-:+                            * 
_read_only_resource_inputs
 *3
config_proto#!

CPU

GPU2*0,1,2,3J 8*R
fMRK
I__inference_activation_16_layer_call_and_return_conditional_losses_2548092
activation_16/PartitionedCall║
!conv2d_14/StatefulPartitionedCallStatefulPartitionedCall&activation_16/PartitionedCall:output:0conv2d_14_254817conv2d_14_254819*
Tin
2*
Tout
2*A
_output_shapes/
-:+                            *$
_read_only_resource_inputs
*3
config_proto#!

CPU

GPU2*0,1,2,3J 8*N
fIRG
E__inference_conv2d_14_layer_call_and_return_conditional_losses_2538142#
!conv2d_14/StatefulPartitionedCall┐
.batch_normalization_13/StatefulPartitionedCallStatefulPartitionedCall*conv2d_14/StatefulPartitionedCall:output:0batch_normalization_13_254848batch_normalization_13_254850batch_normalization_13_254852batch_normalization_13_254854*
Tin	
2*
Tout
2*A
_output_shapes/
-:+                            *$
_read_only_resource_inputs
*3
config_proto#!

CPU

GPU2*0,1,2,3J 8*[
fVRT
R__inference_batch_normalization_13_layer_call_and_return_conditional_losses_25390820
.batch_normalization_13/StatefulPartitionedCallЋ
activation_17/PartitionedCallPartitionedCall7batch_normalization_13/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*A
_output_shapes/
-:+                            * 
_read_only_resource_inputs
 *3
config_proto#!

CPU

GPU2*0,1,2,3J 8*R
fMRK
I__inference_activation_17_layer_call_and_return_conditional_losses_2548622
activation_17/PartitionedCallЋ
add_6/PartitionedCallPartitionedCall&activation_16/PartitionedCall:output:0&activation_17/PartitionedCall:output:0*
Tin
2*
Tout
2*A
_output_shapes/
-:+                            * 
_read_only_resource_inputs
 *3
config_proto#!

CPU

GPU2*0,1,2,3J 8*J
fERC
A__inference_add_6_layer_call_and_return_conditional_losses_2548762
add_6/PartitionedCall▓
!conv2d_15/StatefulPartitionedCallStatefulPartitionedCalladd_6/PartitionedCall:output:0conv2d_15_254885conv2d_15_254887*
Tin
2*
Tout
2*A
_output_shapes/
-:+                            *$
_read_only_resource_inputs
*3
config_proto#!

CPU

GPU2*0,1,2,3J 8*N
fIRG
E__inference_conv2d_15_layer_call_and_return_conditional_losses_2539612#
!conv2d_15/StatefulPartitionedCall┐
.batch_normalization_14/StatefulPartitionedCallStatefulPartitionedCall*conv2d_15/StatefulPartitionedCall:output:0batch_normalization_14_254916batch_normalization_14_254918batch_normalization_14_254920batch_normalization_14_254922*
Tin	
2*
Tout
2*A
_output_shapes/
-:+                            *$
_read_only_resource_inputs
*3
config_proto#!

CPU

GPU2*0,1,2,3J 8*[
fVRT
R__inference_batch_normalization_14_layer_call_and_return_conditional_losses_25405520
.batch_normalization_14/StatefulPartitionedCallЋ
activation_18/PartitionedCallPartitionedCall7batch_normalization_14/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*A
_output_shapes/
-:+                            * 
_read_only_resource_inputs
 *3
config_proto#!

CPU

GPU2*0,1,2,3J 8*R
fMRK
I__inference_activation_18_layer_call_and_return_conditional_losses_2549302
activation_18/PartitionedCallЇ
add_7/PartitionedCallPartitionedCalladd_6/PartitionedCall:output:0&activation_18/PartitionedCall:output:0*
Tin
2*
Tout
2*A
_output_shapes/
-:+                            * 
_read_only_resource_inputs
 *3
config_proto#!

CPU

GPU2*0,1,2,3J 8*J
fERC
A__inference_add_7_layer_call_and_return_conditional_losses_2549442
add_7/PartitionedCallЧ
activation_19/PartitionedCallPartitionedCalladd_7/PartitionedCall:output:0*
Tin
2*
Tout
2*A
_output_shapes/
-:+                            * 
_read_only_resource_inputs
 *3
config_proto#!

CPU

GPU2*0,1,2,3J 8*R
fMRK
I__inference_activation_19_layer_call_and_return_conditional_losses_2549582
activation_19/PartitionedCallі
up_sampling2d_3/PartitionedCallPartitionedCall&activation_19/PartitionedCall:output:0*
Tin
2*
Tout
2*A
_output_shapes/
-:+                            * 
_read_only_resource_inputs
 *3
config_proto#!

CPU

GPU2*0,1,2,3J 8*T
fORM
K__inference_up_sampling2d_3_layer_call_and_return_conditional_losses_2541102!
up_sampling2d_3/PartitionedCall╝
!conv2d_16/StatefulPartitionedCallStatefulPartitionedCall(up_sampling2d_3/PartitionedCall:output:0conv2d_16_254967conv2d_16_254969*
Tin
2*
Tout
2*A
_output_shapes/
-:+                            *$
_read_only_resource_inputs
*3
config_proto#!

CPU

GPU2*0,1,2,3J 8*N
fIRG
E__inference_conv2d_16_layer_call_and_return_conditional_losses_2541272#
!conv2d_16/StatefulPartitionedCall┐
.batch_normalization_15/StatefulPartitionedCallStatefulPartitionedCall*conv2d_16/StatefulPartitionedCall:output:0batch_normalization_15_254998batch_normalization_15_255000batch_normalization_15_255002batch_normalization_15_255004*
Tin	
2*
Tout
2*A
_output_shapes/
-:+                            *$
_read_only_resource_inputs
*3
config_proto#!

CPU

GPU2*0,1,2,3J 8*[
fVRT
R__inference_batch_normalization_15_layer_call_and_return_conditional_losses_25422120
.batch_normalization_15/StatefulPartitionedCallЋ
activation_20/PartitionedCallPartitionedCall7batch_normalization_15/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*A
_output_shapes/
-:+                            * 
_read_only_resource_inputs
 *3
config_proto#!

CPU

GPU2*0,1,2,3J 8*R
fMRK
I__inference_activation_20_layer_call_and_return_conditional_losses_2550122
activation_20/PartitionedCall║
!conv2d_17/StatefulPartitionedCallStatefulPartitionedCall&activation_20/PartitionedCall:output:0conv2d_17_255020conv2d_17_255022*
Tin
2*
Tout
2*A
_output_shapes/
-:+                           *$
_read_only_resource_inputs
*3
config_proto#!

CPU

GPU2*0,1,2,3J 8*N
fIRG
E__inference_conv2d_17_layer_call_and_return_conditional_losses_2542742#
!conv2d_17/StatefulPartitionedCall┐
.batch_normalization_16/StatefulPartitionedCallStatefulPartitionedCall*conv2d_17/StatefulPartitionedCall:output:0batch_normalization_16_255051batch_normalization_16_255053batch_normalization_16_255055batch_normalization_16_255057*
Tin	
2*
Tout
2*A
_output_shapes/
-:+                           *$
_read_only_resource_inputs
*3
config_proto#!

CPU

GPU2*0,1,2,3J 8*[
fVRT
R__inference_batch_normalization_16_layer_call_and_return_conditional_losses_25436820
.batch_normalization_16/StatefulPartitionedCallЋ
activation_21/PartitionedCallPartitionedCall7batch_normalization_16/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*A
_output_shapes/
-:+                           * 
_read_only_resource_inputs
 *3
config_proto#!

CPU

GPU2*0,1,2,3J 8*R
fMRK
I__inference_activation_21_layer_call_and_return_conditional_losses_2550652
activation_21/PartitionedCall╗
IdentityIdentity&activation_21/PartitionedCall:output:0/^batch_normalization_10/StatefulPartitionedCall/^batch_normalization_11/StatefulPartitionedCall/^batch_normalization_12/StatefulPartitionedCall/^batch_normalization_13/StatefulPartitionedCall/^batch_normalization_14/StatefulPartitionedCall/^batch_normalization_15/StatefulPartitionedCall/^batch_normalization_16/StatefulPartitionedCall.^batch_normalization_9/StatefulPartitionedCall"^conv2d_10/StatefulPartitionedCall"^conv2d_11/StatefulPartitionedCall"^conv2d_12/StatefulPartitionedCall"^conv2d_13/StatefulPartitionedCall"^conv2d_14/StatefulPartitionedCall"^conv2d_15/StatefulPartitionedCall"^conv2d_16/StatefulPartitionedCall"^conv2d_17/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*­
_input_shapesя
█:         ::::::::::::::::::::::::::::::::::::::::::::::::2`
.batch_normalization_10/StatefulPartitionedCall.batch_normalization_10/StatefulPartitionedCall2`
.batch_normalization_11/StatefulPartitionedCall.batch_normalization_11/StatefulPartitionedCall2`
.batch_normalization_12/StatefulPartitionedCall.batch_normalization_12/StatefulPartitionedCall2`
.batch_normalization_13/StatefulPartitionedCall.batch_normalization_13/StatefulPartitionedCall2`
.batch_normalization_14/StatefulPartitionedCall.batch_normalization_14/StatefulPartitionedCall2`
.batch_normalization_15/StatefulPartitionedCall.batch_normalization_15/StatefulPartitionedCall2`
.batch_normalization_16/StatefulPartitionedCall.batch_normalization_16/StatefulPartitionedCall2^
-batch_normalization_9/StatefulPartitionedCall-batch_normalization_9/StatefulPartitionedCall2F
!conv2d_10/StatefulPartitionedCall!conv2d_10/StatefulPartitionedCall2F
!conv2d_11/StatefulPartitionedCall!conv2d_11/StatefulPartitionedCall2F
!conv2d_12/StatefulPartitionedCall!conv2d_12/StatefulPartitionedCall2F
!conv2d_13/StatefulPartitionedCall!conv2d_13/StatefulPartitionedCall2F
!conv2d_14/StatefulPartitionedCall!conv2d_14/StatefulPartitionedCall2F
!conv2d_15/StatefulPartitionedCall!conv2d_15/StatefulPartitionedCall2F
!conv2d_16/StatefulPartitionedCall!conv2d_16/StatefulPartitionedCall2F
!conv2d_17/StatefulPartitionedCall!conv2d_17/StatefulPartitionedCall:X T
/
_output_shapes
:         
!
_user_specified_name	input_3:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
: :!

_output_shapes
: :"

_output_shapes
: :#

_output_shapes
: :$

_output_shapes
: :%

_output_shapes
: :&

_output_shapes
: :'

_output_shapes
: :(

_output_shapes
: :)

_output_shapes
: :*

_output_shapes
: :+

_output_shapes
: :,

_output_shapes
: :-

_output_shapes
: :.

_output_shapes
: :/

_output_shapes
: :0

_output_shapes
: 
В
Л
$__inference_signature_wrapper_255872
input_3
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34

unknown_35

unknown_36

unknown_37

unknown_38

unknown_39

unknown_40

unknown_41

unknown_42

unknown_43

unknown_44

unknown_45

unknown_46
identityѕбStatefulPartitionedCall╝
StatefulPartitionedCallStatefulPartitionedCallinput_3unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46*<
Tin5
321*
Tout
2*/
_output_shapes
:         *R
_read_only_resource_inputs4
20	
 !"#$%&'()*+,-./0*3
config_proto#!

CPU

GPU2*0,1,2,3J 8**
f%R#
!__inference__wrapped_model_2531962
StatefulPartitionedCallќ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:         2

Identity"
identityIdentity:output:0*­
_input_shapesя
█:         ::::::::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:         
!
_user_specified_name	input_3:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
: :!

_output_shapes
: :"

_output_shapes
: :#

_output_shapes
: :$

_output_shapes
: :%

_output_shapes
: :&

_output_shapes
: :'

_output_shapes
: :(

_output_shapes
: :)

_output_shapes
: :*

_output_shapes
: :+

_output_shapes
: :,

_output_shapes
: :-

_output_shapes
: :.

_output_shapes
: :/

_output_shapes
: :0

_output_shapes
: 
И
ф
7__inference_batch_normalization_10_layer_call_fn_256825

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityѕбStatefulPartitionedCallІ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*/
_output_shapes
:          *&
_read_only_resource_inputs
*3
config_proto#!

CPU

GPU2*0,1,2,3J 8*[
fVRT
R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_2545692
StatefulPartitionedCallќ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:          2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:          ::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:          
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
ђ	
ф
7__inference_batch_normalization_15_layer_call_fn_257528

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityѕбStatefulPartitionedCallЮ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*A
_output_shapes/
-:+                            *&
_read_only_resource_inputs
*3
config_proto#!

CPU

GPU2*0,1,2,3J 8*[
fVRT
R__inference_batch_normalization_15_layer_call_and_return_conditional_losses_2542522
StatefulPartitionedCallе
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                            2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                            ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Б
J
.__inference_activation_13_layer_call_fn_256910

inputs
identityХ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*/
_output_shapes
:          * 
_read_only_resource_inputs
 *3
config_proto#!

CPU

GPU2*0,1,2,3J 8*R
fMRK
I__inference_activation_13_layer_call_and_return_conditional_losses_2546102
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:          2

Identity"
identityIdentity:output:0*.
_input_shapes
:          :W S
/
_output_shapes
:          
 
_user_specified_nameinputs
а
e
I__inference_activation_17_layer_call_and_return_conditional_losses_257305

inputs
identityh
ReluReluinputs*
T0*A
_output_shapes/
-:+                            2
Reluђ
IdentityIdentityRelu:activations:0*
T0*A
_output_shapes/
-:+                            2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+                            :i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
а
e
I__inference_activation_18_layer_call_and_return_conditional_losses_257414

inputs
identityh
ReluReluinputs*
T0*A
_output_shapes/
-:+                            2
Reluђ
IdentityIdentityRelu:activations:0*
T0*A
_output_shapes/
-:+                            2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+                            :i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
е
R
&__inference_add_4_layer_call_fn_256922
inputs_0
inputs_1
identity╗
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*/
_output_shapes
:          * 
_read_only_resource_inputs
 *3
config_proto#!

CPU

GPU2*0,1,2,3J 8*J
fERC
A__inference_add_4_layer_call_and_return_conditional_losses_2546242
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:          2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:          :          :Y U
/
_output_shapes
:          
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:          
"
_user_specified_name
inputs/1
Б
J
.__inference_activation_14_layer_call_fn_257094

inputs
identityХ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*/
_output_shapes
:          * 
_read_only_resource_inputs
 *3
config_proto#!

CPU

GPU2*0,1,2,3J 8*R
fMRK
I__inference_activation_14_layer_call_and_return_conditional_losses_2547272
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:          2

Identity"
identityIdentity:output:0*.
_input_shapes
:          :W S
/
_output_shapes
:          
 
_user_specified_nameinputs
њ
І
R__inference_batch_normalization_15_layer_call_and_return_conditional_losses_257502

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityѕt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOpГ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1▄
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                            : : : : :*
epsilon%oЃ:*
is_training( 2
FusedBatchNormV3ѓ
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+                            2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                            :::::i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
О
m
A__inference_add_5_layer_call_and_return_conditional_losses_257100
inputs_0
inputs_1
identitya
addAddV2inputs_0inputs_1*
T0*/
_output_shapes
:          2
addc
IdentityIdentityadd:z:0*
T0*/
_output_shapes
:          2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:          :          :Y U
/
_output_shapes
:          
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:          
"
_user_specified_name
inputs/1
■
ф
7__inference_batch_normalization_12_layer_call_fn_257190

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityѕбStatefulPartitionedCallЏ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*A
_output_shapes/
-:+                            *$
_read_only_resource_inputs
*3
config_proto#!

CPU

GPU2*0,1,2,3J 8*[
fVRT
R__inference_batch_normalization_12_layer_call_and_return_conditional_losses_2537612
StatefulPartitionedCallе
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                            2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                            ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
њ
І
R__inference_batch_normalization_13_layer_call_and_return_conditional_losses_253939

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityѕt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOpГ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1▄
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                            : : : : :*
epsilon%oЃ:*
is_training( 2
FusedBatchNormV3ѓ
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+                            2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                            :::::i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
В
J
.__inference_activation_20_layer_call_fn_257538

inputs
identity╚
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*A
_output_shapes/
-:+                            * 
_read_only_resource_inputs
 *3
config_proto#!

CPU

GPU2*0,1,2,3J 8*R
fMRK
I__inference_activation_20_layer_call_and_return_conditional_losses_2550122
PartitionedCallє
IdentityIdentityPartitionedCall:output:0*
T0*A
_output_shapes/
-:+                            2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+                            :i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
╝	
Г
E__inference_conv2d_10_layer_call_and_return_conditional_losses_253207

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityѕЋ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOpх
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                            *
paddingSAME*
strides
2
Conv2Dї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOpџ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                            2	
BiasAdd~
IdentityIdentityBiasAdd:output:0*
T0*A
_output_shapes/
-:+                            2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+                           :::i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
У$
┘
R__inference_batch_normalization_12_layer_call_and_return_conditional_losses_253761

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityѕб#AssignMovingAvg/AssignSubVariableOpб%AssignMovingAvg_1/AssignSubVariableOpt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOpГ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1╔
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                            : : : : :*
epsilon%oЃ:2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
Const░
AssignMovingAvg/sub/xConst*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  ђ?2
AssignMovingAvg/sub/x┐
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const:output:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/subЦ
AssignMovingAvg/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOpя
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/sub_1К
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/mulК
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp(fusedbatchnormv3_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpХ
AssignMovingAvg_1/sub/xConst*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  ђ?2
AssignMovingAvg_1/sub/xК
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const:output:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/subФ
 AssignMovingAvg_1/ReadVariableOpReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOpЖ
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/sub_1Л
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/mulН
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp*fusedbatchnormv3_readvariableop_1_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpл
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*A
_output_shapes/
-:+                            2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                            ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
у$
п
Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_256609

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityѕб#AssignMovingAvg/AssignSubVariableOpб%AssignMovingAvg_1/AssignSubVariableOpt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOpГ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1╔
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                            : : : : :*
epsilon%oЃ:2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
Const░
AssignMovingAvg/sub/xConst*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  ђ?2
AssignMovingAvg/sub/x┐
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const:output:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/subЦ
AssignMovingAvg/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOpя
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/sub_1К
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/mulК
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp(fusedbatchnormv3_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpХ
AssignMovingAvg_1/sub/xConst*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  ђ?2
AssignMovingAvg_1/sub/xК
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const:output:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/subФ
 AssignMovingAvg_1/ReadVariableOpReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOpЖ
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/sub_1Л
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/mulН
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp*fusedbatchnormv3_readvariableop_1_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpл
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*A
_output_shapes/
-:+                            2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                            ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
╝	
Г
E__inference_conv2d_17_layer_call_and_return_conditional_losses_254274

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityѕЋ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOpх
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           *
paddingSAME*
strides
2
Conv2Dї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpџ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           2	
BiasAdd~
IdentityIdentityBiasAdd:output:0*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+                            :::i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
Ъ$
п
Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_256684

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityѕб#AssignMovingAvg/AssignSubVariableOpб%AssignMovingAvg_1/AssignSubVariableOpt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOpГ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1и
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:          : : : : :*
epsilon%oЃ:2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
Const░
AssignMovingAvg/sub/xConst*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  ђ?2
AssignMovingAvg/sub/x┐
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const:output:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/subЦ
AssignMovingAvg/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOpя
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/sub_1К
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/mulК
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp(fusedbatchnormv3_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpХ
AssignMovingAvg_1/sub/xConst*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  ђ?2
AssignMovingAvg_1/sub/xК
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const:output:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/subФ
 AssignMovingAvg_1/ReadVariableOpReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOpЖ
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/sub_1Л
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/mulН
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp*fusedbatchnormv3_readvariableop_1_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpЙ
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*/
_output_shapes
:          2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:          ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:W S
/
_output_shapes
:          
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
а$
┘
R__inference_batch_normalization_11_layer_call_and_return_conditional_losses_256965

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityѕб#AssignMovingAvg/AssignSubVariableOpб%AssignMovingAvg_1/AssignSubVariableOpt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOpГ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1и
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:          : : : : :*
epsilon%oЃ:2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
Const░
AssignMovingAvg/sub/xConst*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  ђ?2
AssignMovingAvg/sub/x┐
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const:output:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/subЦ
AssignMovingAvg/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOpя
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/sub_1К
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/mulК
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp(fusedbatchnormv3_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpХ
AssignMovingAvg_1/sub/xConst*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  ђ?2
AssignMovingAvg_1/sub/xК
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const:output:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/subФ
 AssignMovingAvg_1/ReadVariableOpReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOpЖ
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/sub_1Л
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/mulН
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp*fusedbatchnormv3_readvariableop_1_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpЙ
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*/
_output_shapes
:          2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:          ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:W S
/
_output_shapes
:          
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
ж

*__inference_conv2d_17_layer_call_fn_254284

inputs
unknown
	unknown_0
identityѕбStatefulPartitionedCallШ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*A
_output_shapes/
-:+                           *$
_read_only_resource_inputs
*3
config_proto#!

CPU

GPU2*0,1,2,3J 8*N
fIRG
E__inference_conv2d_17_layer_call_and_return_conditional_losses_2542742
StatefulPartitionedCallе
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+                            ::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
├
m
A__inference_add_6_layer_call_and_return_conditional_losses_257316
inputs_0
inputs_1
identitys
addAddV2inputs_0inputs_1*
T0*A
_output_shapes/
-:+                            2
addu
IdentityIdentityadd:z:0*
T0*A
_output_shapes/
-:+                            2

Identity"
identityIdentity:output:0*m
_input_shapes\
Z:+                            :+                            :k g
A
_output_shapes/
-:+                            
"
_user_specified_name
inputs/0:kg
A
_output_shapes/
-:+                            
"
_user_specified_name
inputs/1
а
e
I__inference_activation_20_layer_call_and_return_conditional_losses_255012

inputs
identityh
ReluReluinputs*
T0*A
_output_shapes/
-:+                            2
Reluђ
IdentityIdentityRelu:activations:0*
T0*A
_output_shapes/
-:+                            2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+                            :i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
за
┼
C__inference_model_2_layer_call_and_return_conditional_losses_255341

inputs
conv2d_10_255212
conv2d_10_255214 
batch_normalization_9_255217 
batch_normalization_9_255219 
batch_normalization_9_255221 
batch_normalization_9_255223
conv2d_11_255227
conv2d_11_255229!
batch_normalization_10_255232!
batch_normalization_10_255234!
batch_normalization_10_255236!
batch_normalization_10_255238
conv2d_12_255243
conv2d_12_255245!
batch_normalization_11_255248!
batch_normalization_11_255250!
batch_normalization_11_255252!
batch_normalization_11_255254
conv2d_13_255261
conv2d_13_255263!
batch_normalization_12_255266!
batch_normalization_12_255268!
batch_normalization_12_255270!
batch_normalization_12_255272
conv2d_14_255276
conv2d_14_255278!
batch_normalization_13_255281!
batch_normalization_13_255283!
batch_normalization_13_255285!
batch_normalization_13_255287
conv2d_15_255292
conv2d_15_255294!
batch_normalization_14_255297!
batch_normalization_14_255299!
batch_normalization_14_255301!
batch_normalization_14_255303
conv2d_16_255310
conv2d_16_255312!
batch_normalization_15_255315!
batch_normalization_15_255317!
batch_normalization_15_255319!
batch_normalization_15_255321
conv2d_17_255325
conv2d_17_255327!
batch_normalization_16_255330!
batch_normalization_16_255332!
batch_normalization_16_255334!
batch_normalization_16_255336
identityѕб.batch_normalization_10/StatefulPartitionedCallб.batch_normalization_11/StatefulPartitionedCallб.batch_normalization_12/StatefulPartitionedCallб.batch_normalization_13/StatefulPartitionedCallб.batch_normalization_14/StatefulPartitionedCallб.batch_normalization_15/StatefulPartitionedCallб.batch_normalization_16/StatefulPartitionedCallб-batch_normalization_9/StatefulPartitionedCallб!conv2d_10/StatefulPartitionedCallб!conv2d_11/StatefulPartitionedCallб!conv2d_12/StatefulPartitionedCallб!conv2d_13/StatefulPartitionedCallб!conv2d_14/StatefulPartitionedCallб!conv2d_15/StatefulPartitionedCallб!conv2d_16/StatefulPartitionedCallб!conv2d_17/StatefulPartitionedCallѕ
!conv2d_10/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_10_255212conv2d_10_255214*
Tin
2*
Tout
2*/
_output_shapes
:          *$
_read_only_resource_inputs
*3
config_proto#!

CPU

GPU2*0,1,2,3J 8*N
fIRG
E__inference_conv2d_10_layer_call_and_return_conditional_losses_2532072#
!conv2d_10/StatefulPartitionedCallд
-batch_normalization_9/StatefulPartitionedCallStatefulPartitionedCall*conv2d_10/StatefulPartitionedCall:output:0batch_normalization_9_255217batch_normalization_9_255219batch_normalization_9_255221batch_normalization_9_255223*
Tin	
2*
Tout
2*/
_output_shapes
:          *$
_read_only_resource_inputs
*3
config_proto#!

CPU

GPU2*0,1,2,3J 8*Z
fURS
Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_2544492/
-batch_normalization_9/StatefulPartitionedCallѓ
activation_12/PartitionedCallPartitionedCall6batch_normalization_9/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:          * 
_read_only_resource_inputs
 *3
config_proto#!

CPU

GPU2*0,1,2,3J 8*R
fMRK
I__inference_activation_12_layer_call_and_return_conditional_losses_2545082
activation_12/PartitionedCallе
!conv2d_11/StatefulPartitionedCallStatefulPartitionedCall&activation_12/PartitionedCall:output:0conv2d_11_255227conv2d_11_255229*
Tin
2*
Tout
2*/
_output_shapes
:          *$
_read_only_resource_inputs
*3
config_proto#!

CPU

GPU2*0,1,2,3J 8*N
fIRG
E__inference_conv2d_11_layer_call_and_return_conditional_losses_2533542#
!conv2d_11/StatefulPartitionedCallГ
.batch_normalization_10/StatefulPartitionedCallStatefulPartitionedCall*conv2d_11/StatefulPartitionedCall:output:0batch_normalization_10_255232batch_normalization_10_255234batch_normalization_10_255236batch_normalization_10_255238*
Tin	
2*
Tout
2*/
_output_shapes
:          *$
_read_only_resource_inputs
*3
config_proto#!

CPU

GPU2*0,1,2,3J 8*[
fVRT
R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_25455120
.batch_normalization_10/StatefulPartitionedCallЃ
activation_13/PartitionedCallPartitionedCall7batch_normalization_10/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:          * 
_read_only_resource_inputs
 *3
config_proto#!

CPU

GPU2*0,1,2,3J 8*R
fMRK
I__inference_activation_13_layer_call_and_return_conditional_losses_2546102
activation_13/PartitionedCallЃ
add_4/PartitionedCallPartitionedCall&activation_12/PartitionedCall:output:0&activation_13/PartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:          * 
_read_only_resource_inputs
 *3
config_proto#!

CPU

GPU2*0,1,2,3J 8*J
fERC
A__inference_add_4_layer_call_and_return_conditional_losses_2546242
add_4/PartitionedCallа
!conv2d_12/StatefulPartitionedCallStatefulPartitionedCalladd_4/PartitionedCall:output:0conv2d_12_255243conv2d_12_255245*
Tin
2*
Tout
2*/
_output_shapes
:          *$
_read_only_resource_inputs
*3
config_proto#!

CPU

GPU2*0,1,2,3J 8*N
fIRG
E__inference_conv2d_12_layer_call_and_return_conditional_losses_2535012#
!conv2d_12/StatefulPartitionedCallГ
.batch_normalization_11/StatefulPartitionedCallStatefulPartitionedCall*conv2d_12/StatefulPartitionedCall:output:0batch_normalization_11_255248batch_normalization_11_255250batch_normalization_11_255252batch_normalization_11_255254*
Tin	
2*
Tout
2*/
_output_shapes
:          *$
_read_only_resource_inputs
*3
config_proto#!

CPU

GPU2*0,1,2,3J 8*[
fVRT
R__inference_batch_normalization_11_layer_call_and_return_conditional_losses_25466820
.batch_normalization_11/StatefulPartitionedCallЃ
activation_14/PartitionedCallPartitionedCall7batch_normalization_11/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:          * 
_read_only_resource_inputs
 *3
config_proto#!

CPU

GPU2*0,1,2,3J 8*R
fMRK
I__inference_activation_14_layer_call_and_return_conditional_losses_2547272
activation_14/PartitionedCallч
add_5/PartitionedCallPartitionedCalladd_4/PartitionedCall:output:0&activation_14/PartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:          * 
_read_only_resource_inputs
 *3
config_proto#!

CPU

GPU2*0,1,2,3J 8*J
fERC
A__inference_add_5_layer_call_and_return_conditional_losses_2547412
add_5/PartitionedCallЖ
activation_15/PartitionedCallPartitionedCalladd_5/PartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:          * 
_read_only_resource_inputs
 *3
config_proto#!

CPU

GPU2*0,1,2,3J 8*R
fMRK
I__inference_activation_15_layer_call_and_return_conditional_losses_2547552
activation_15/PartitionedCallі
up_sampling2d_2/PartitionedCallPartitionedCall&activation_15/PartitionedCall:output:0*
Tin
2*
Tout
2*A
_output_shapes/
-:+                            * 
_read_only_resource_inputs
 *3
config_proto#!

CPU

GPU2*0,1,2,3J 8*T
fORM
K__inference_up_sampling2d_2_layer_call_and_return_conditional_losses_2536502!
up_sampling2d_2/PartitionedCall╝
!conv2d_13/StatefulPartitionedCallStatefulPartitionedCall(up_sampling2d_2/PartitionedCall:output:0conv2d_13_255261conv2d_13_255263*
Tin
2*
Tout
2*A
_output_shapes/
-:+                            *$
_read_only_resource_inputs
*3
config_proto#!

CPU

GPU2*0,1,2,3J 8*N
fIRG
E__inference_conv2d_13_layer_call_and_return_conditional_losses_2536672#
!conv2d_13/StatefulPartitionedCall┐
.batch_normalization_12/StatefulPartitionedCallStatefulPartitionedCall*conv2d_13/StatefulPartitionedCall:output:0batch_normalization_12_255266batch_normalization_12_255268batch_normalization_12_255270batch_normalization_12_255272*
Tin	
2*
Tout
2*A
_output_shapes/
-:+                            *$
_read_only_resource_inputs
*3
config_proto#!

CPU

GPU2*0,1,2,3J 8*[
fVRT
R__inference_batch_normalization_12_layer_call_and_return_conditional_losses_25376120
.batch_normalization_12/StatefulPartitionedCallЋ
activation_16/PartitionedCallPartitionedCall7batch_normalization_12/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*A
_output_shapes/
-:+                            * 
_read_only_resource_inputs
 *3
config_proto#!

CPU

GPU2*0,1,2,3J 8*R
fMRK
I__inference_activation_16_layer_call_and_return_conditional_losses_2548092
activation_16/PartitionedCall║
!conv2d_14/StatefulPartitionedCallStatefulPartitionedCall&activation_16/PartitionedCall:output:0conv2d_14_255276conv2d_14_255278*
Tin
2*
Tout
2*A
_output_shapes/
-:+                            *$
_read_only_resource_inputs
*3
config_proto#!

CPU

GPU2*0,1,2,3J 8*N
fIRG
E__inference_conv2d_14_layer_call_and_return_conditional_losses_2538142#
!conv2d_14/StatefulPartitionedCall┐
.batch_normalization_13/StatefulPartitionedCallStatefulPartitionedCall*conv2d_14/StatefulPartitionedCall:output:0batch_normalization_13_255281batch_normalization_13_255283batch_normalization_13_255285batch_normalization_13_255287*
Tin	
2*
Tout
2*A
_output_shapes/
-:+                            *$
_read_only_resource_inputs
*3
config_proto#!

CPU

GPU2*0,1,2,3J 8*[
fVRT
R__inference_batch_normalization_13_layer_call_and_return_conditional_losses_25390820
.batch_normalization_13/StatefulPartitionedCallЋ
activation_17/PartitionedCallPartitionedCall7batch_normalization_13/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*A
_output_shapes/
-:+                            * 
_read_only_resource_inputs
 *3
config_proto#!

CPU

GPU2*0,1,2,3J 8*R
fMRK
I__inference_activation_17_layer_call_and_return_conditional_losses_2548622
activation_17/PartitionedCallЋ
add_6/PartitionedCallPartitionedCall&activation_16/PartitionedCall:output:0&activation_17/PartitionedCall:output:0*
Tin
2*
Tout
2*A
_output_shapes/
-:+                            * 
_read_only_resource_inputs
 *3
config_proto#!

CPU

GPU2*0,1,2,3J 8*J
fERC
A__inference_add_6_layer_call_and_return_conditional_losses_2548762
add_6/PartitionedCall▓
!conv2d_15/StatefulPartitionedCallStatefulPartitionedCalladd_6/PartitionedCall:output:0conv2d_15_255292conv2d_15_255294*
Tin
2*
Tout
2*A
_output_shapes/
-:+                            *$
_read_only_resource_inputs
*3
config_proto#!

CPU

GPU2*0,1,2,3J 8*N
fIRG
E__inference_conv2d_15_layer_call_and_return_conditional_losses_2539612#
!conv2d_15/StatefulPartitionedCall┐
.batch_normalization_14/StatefulPartitionedCallStatefulPartitionedCall*conv2d_15/StatefulPartitionedCall:output:0batch_normalization_14_255297batch_normalization_14_255299batch_normalization_14_255301batch_normalization_14_255303*
Tin	
2*
Tout
2*A
_output_shapes/
-:+                            *$
_read_only_resource_inputs
*3
config_proto#!

CPU

GPU2*0,1,2,3J 8*[
fVRT
R__inference_batch_normalization_14_layer_call_and_return_conditional_losses_25405520
.batch_normalization_14/StatefulPartitionedCallЋ
activation_18/PartitionedCallPartitionedCall7batch_normalization_14/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*A
_output_shapes/
-:+                            * 
_read_only_resource_inputs
 *3
config_proto#!

CPU

GPU2*0,1,2,3J 8*R
fMRK
I__inference_activation_18_layer_call_and_return_conditional_losses_2549302
activation_18/PartitionedCallЇ
add_7/PartitionedCallPartitionedCalladd_6/PartitionedCall:output:0&activation_18/PartitionedCall:output:0*
Tin
2*
Tout
2*A
_output_shapes/
-:+                            * 
_read_only_resource_inputs
 *3
config_proto#!

CPU

GPU2*0,1,2,3J 8*J
fERC
A__inference_add_7_layer_call_and_return_conditional_losses_2549442
add_7/PartitionedCallЧ
activation_19/PartitionedCallPartitionedCalladd_7/PartitionedCall:output:0*
Tin
2*
Tout
2*A
_output_shapes/
-:+                            * 
_read_only_resource_inputs
 *3
config_proto#!

CPU

GPU2*0,1,2,3J 8*R
fMRK
I__inference_activation_19_layer_call_and_return_conditional_losses_2549582
activation_19/PartitionedCallі
up_sampling2d_3/PartitionedCallPartitionedCall&activation_19/PartitionedCall:output:0*
Tin
2*
Tout
2*A
_output_shapes/
-:+                            * 
_read_only_resource_inputs
 *3
config_proto#!

CPU

GPU2*0,1,2,3J 8*T
fORM
K__inference_up_sampling2d_3_layer_call_and_return_conditional_losses_2541102!
up_sampling2d_3/PartitionedCall╝
!conv2d_16/StatefulPartitionedCallStatefulPartitionedCall(up_sampling2d_3/PartitionedCall:output:0conv2d_16_255310conv2d_16_255312*
Tin
2*
Tout
2*A
_output_shapes/
-:+                            *$
_read_only_resource_inputs
*3
config_proto#!

CPU

GPU2*0,1,2,3J 8*N
fIRG
E__inference_conv2d_16_layer_call_and_return_conditional_losses_2541272#
!conv2d_16/StatefulPartitionedCall┐
.batch_normalization_15/StatefulPartitionedCallStatefulPartitionedCall*conv2d_16/StatefulPartitionedCall:output:0batch_normalization_15_255315batch_normalization_15_255317batch_normalization_15_255319batch_normalization_15_255321*
Tin	
2*
Tout
2*A
_output_shapes/
-:+                            *$
_read_only_resource_inputs
*3
config_proto#!

CPU

GPU2*0,1,2,3J 8*[
fVRT
R__inference_batch_normalization_15_layer_call_and_return_conditional_losses_25422120
.batch_normalization_15/StatefulPartitionedCallЋ
activation_20/PartitionedCallPartitionedCall7batch_normalization_15/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*A
_output_shapes/
-:+                            * 
_read_only_resource_inputs
 *3
config_proto#!

CPU

GPU2*0,1,2,3J 8*R
fMRK
I__inference_activation_20_layer_call_and_return_conditional_losses_2550122
activation_20/PartitionedCall║
!conv2d_17/StatefulPartitionedCallStatefulPartitionedCall&activation_20/PartitionedCall:output:0conv2d_17_255325conv2d_17_255327*
Tin
2*
Tout
2*A
_output_shapes/
-:+                           *$
_read_only_resource_inputs
*3
config_proto#!

CPU

GPU2*0,1,2,3J 8*N
fIRG
E__inference_conv2d_17_layer_call_and_return_conditional_losses_2542742#
!conv2d_17/StatefulPartitionedCall┐
.batch_normalization_16/StatefulPartitionedCallStatefulPartitionedCall*conv2d_17/StatefulPartitionedCall:output:0batch_normalization_16_255330batch_normalization_16_255332batch_normalization_16_255334batch_normalization_16_255336*
Tin	
2*
Tout
2*A
_output_shapes/
-:+                           *$
_read_only_resource_inputs
*3
config_proto#!

CPU

GPU2*0,1,2,3J 8*[
fVRT
R__inference_batch_normalization_16_layer_call_and_return_conditional_losses_25436820
.batch_normalization_16/StatefulPartitionedCallЋ
activation_21/PartitionedCallPartitionedCall7batch_normalization_16/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*A
_output_shapes/
-:+                           * 
_read_only_resource_inputs
 *3
config_proto#!

CPU

GPU2*0,1,2,3J 8*R
fMRK
I__inference_activation_21_layer_call_and_return_conditional_losses_2550652
activation_21/PartitionedCall╗
IdentityIdentity&activation_21/PartitionedCall:output:0/^batch_normalization_10/StatefulPartitionedCall/^batch_normalization_11/StatefulPartitionedCall/^batch_normalization_12/StatefulPartitionedCall/^batch_normalization_13/StatefulPartitionedCall/^batch_normalization_14/StatefulPartitionedCall/^batch_normalization_15/StatefulPartitionedCall/^batch_normalization_16/StatefulPartitionedCall.^batch_normalization_9/StatefulPartitionedCall"^conv2d_10/StatefulPartitionedCall"^conv2d_11/StatefulPartitionedCall"^conv2d_12/StatefulPartitionedCall"^conv2d_13/StatefulPartitionedCall"^conv2d_14/StatefulPartitionedCall"^conv2d_15/StatefulPartitionedCall"^conv2d_16/StatefulPartitionedCall"^conv2d_17/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*­
_input_shapesя
█:         ::::::::::::::::::::::::::::::::::::::::::::::::2`
.batch_normalization_10/StatefulPartitionedCall.batch_normalization_10/StatefulPartitionedCall2`
.batch_normalization_11/StatefulPartitionedCall.batch_normalization_11/StatefulPartitionedCall2`
.batch_normalization_12/StatefulPartitionedCall.batch_normalization_12/StatefulPartitionedCall2`
.batch_normalization_13/StatefulPartitionedCall.batch_normalization_13/StatefulPartitionedCall2`
.batch_normalization_14/StatefulPartitionedCall.batch_normalization_14/StatefulPartitionedCall2`
.batch_normalization_15/StatefulPartitionedCall.batch_normalization_15/StatefulPartitionedCall2`
.batch_normalization_16/StatefulPartitionedCall.batch_normalization_16/StatefulPartitionedCall2^
-batch_normalization_9/StatefulPartitionedCall-batch_normalization_9/StatefulPartitionedCall2F
!conv2d_10/StatefulPartitionedCall!conv2d_10/StatefulPartitionedCall2F
!conv2d_11/StatefulPartitionedCall!conv2d_11/StatefulPartitionedCall2F
!conv2d_12/StatefulPartitionedCall!conv2d_12/StatefulPartitionedCall2F
!conv2d_13/StatefulPartitionedCall!conv2d_13/StatefulPartitionedCall2F
!conv2d_14/StatefulPartitionedCall!conv2d_14/StatefulPartitionedCall2F
!conv2d_15/StatefulPartitionedCall!conv2d_15/StatefulPartitionedCall2F
!conv2d_16/StatefulPartitionedCall!conv2d_16/StatefulPartitionedCall2F
!conv2d_17/StatefulPartitionedCall!conv2d_17/StatefulPartitionedCall:W S
/
_output_shapes
:         
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
: :!

_output_shapes
: :"

_output_shapes
: :#

_output_shapes
: :$

_output_shapes
: :%

_output_shapes
: :&

_output_shapes
: :'

_output_shapes
: :(

_output_shapes
: :)

_output_shapes
: :*

_output_shapes
: :+

_output_shapes
: :,

_output_shapes
: :-

_output_shapes
: :.

_output_shapes
: :/

_output_shapes
: :0

_output_shapes
: 
▀¤
Ш
C__inference_model_2_layer_call_and_return_conditional_losses_256364

inputs,
(conv2d_10_conv2d_readvariableop_resource-
)conv2d_10_biasadd_readvariableop_resource1
-batch_normalization_9_readvariableop_resource3
/batch_normalization_9_readvariableop_1_resourceB
>batch_normalization_9_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource,
(conv2d_11_conv2d_readvariableop_resource-
)conv2d_11_biasadd_readvariableop_resource2
.batch_normalization_10_readvariableop_resource4
0batch_normalization_10_readvariableop_1_resourceC
?batch_normalization_10_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_10_fusedbatchnormv3_readvariableop_1_resource,
(conv2d_12_conv2d_readvariableop_resource-
)conv2d_12_biasadd_readvariableop_resource2
.batch_normalization_11_readvariableop_resource4
0batch_normalization_11_readvariableop_1_resourceC
?batch_normalization_11_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_11_fusedbatchnormv3_readvariableop_1_resource,
(conv2d_13_conv2d_readvariableop_resource-
)conv2d_13_biasadd_readvariableop_resource2
.batch_normalization_12_readvariableop_resource4
0batch_normalization_12_readvariableop_1_resourceC
?batch_normalization_12_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_12_fusedbatchnormv3_readvariableop_1_resource,
(conv2d_14_conv2d_readvariableop_resource-
)conv2d_14_biasadd_readvariableop_resource2
.batch_normalization_13_readvariableop_resource4
0batch_normalization_13_readvariableop_1_resourceC
?batch_normalization_13_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_13_fusedbatchnormv3_readvariableop_1_resource,
(conv2d_15_conv2d_readvariableop_resource-
)conv2d_15_biasadd_readvariableop_resource2
.batch_normalization_14_readvariableop_resource4
0batch_normalization_14_readvariableop_1_resourceC
?batch_normalization_14_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_14_fusedbatchnormv3_readvariableop_1_resource,
(conv2d_16_conv2d_readvariableop_resource-
)conv2d_16_biasadd_readvariableop_resource2
.batch_normalization_15_readvariableop_resource4
0batch_normalization_15_readvariableop_1_resourceC
?batch_normalization_15_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_15_fusedbatchnormv3_readvariableop_1_resource,
(conv2d_17_conv2d_readvariableop_resource-
)conv2d_17_biasadd_readvariableop_resource2
.batch_normalization_16_readvariableop_resource4
0batch_normalization_16_readvariableop_1_resourceC
?batch_normalization_16_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_16_fusedbatchnormv3_readvariableop_1_resource
identityѕ│
conv2d_10/Conv2D/ReadVariableOpReadVariableOp(conv2d_10_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_10/Conv2D/ReadVariableOp┴
conv2d_10/Conv2DConv2Dinputs'conv2d_10/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:          *
paddingSAME*
strides
2
conv2d_10/Conv2Dф
 conv2d_10/BiasAdd/ReadVariableOpReadVariableOp)conv2d_10_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_10/BiasAdd/ReadVariableOp░
conv2d_10/BiasAddBiasAddconv2d_10/Conv2D:output:0(conv2d_10/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          2
conv2d_10/BiasAddХ
$batch_normalization_9/ReadVariableOpReadVariableOp-batch_normalization_9_readvariableop_resource*
_output_shapes
: *
dtype02&
$batch_normalization_9/ReadVariableOp╝
&batch_normalization_9/ReadVariableOp_1ReadVariableOp/batch_normalization_9_readvariableop_1_resource*
_output_shapes
: *
dtype02(
&batch_normalization_9/ReadVariableOp_1ж
5batch_normalization_9/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_9_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype027
5batch_normalization_9/FusedBatchNormV3/ReadVariableOp№
7batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype029
7batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1Р
&batch_normalization_9/FusedBatchNormV3FusedBatchNormV3conv2d_10/BiasAdd:output:0,batch_normalization_9/ReadVariableOp:value:0.batch_normalization_9/ReadVariableOp_1:value:0=batch_normalization_9/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:          : : : : :*
epsilon%oЃ:*
is_training( 2(
&batch_normalization_9/FusedBatchNormV3ќ
activation_12/ReluRelu*batch_normalization_9/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:          2
activation_12/Relu│
conv2d_11/Conv2D/ReadVariableOpReadVariableOp(conv2d_11_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02!
conv2d_11/Conv2D/ReadVariableOp█
conv2d_11/Conv2DConv2D activation_12/Relu:activations:0'conv2d_11/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:          *
paddingSAME*
strides
2
conv2d_11/Conv2Dф
 conv2d_11/BiasAdd/ReadVariableOpReadVariableOp)conv2d_11_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_11/BiasAdd/ReadVariableOp░
conv2d_11/BiasAddBiasAddconv2d_11/Conv2D:output:0(conv2d_11/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          2
conv2d_11/BiasAdd╣
%batch_normalization_10/ReadVariableOpReadVariableOp.batch_normalization_10_readvariableop_resource*
_output_shapes
: *
dtype02'
%batch_normalization_10/ReadVariableOp┐
'batch_normalization_10/ReadVariableOp_1ReadVariableOp0batch_normalization_10_readvariableop_1_resource*
_output_shapes
: *
dtype02)
'batch_normalization_10/ReadVariableOp_1В
6batch_normalization_10/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_10_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype028
6batch_normalization_10/FusedBatchNormV3/ReadVariableOpЫ
8batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_10_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02:
8batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1У
'batch_normalization_10/FusedBatchNormV3FusedBatchNormV3conv2d_11/BiasAdd:output:0-batch_normalization_10/ReadVariableOp:value:0/batch_normalization_10/ReadVariableOp_1:value:0>batch_normalization_10/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:          : : : : :*
epsilon%oЃ:*
is_training( 2)
'batch_normalization_10/FusedBatchNormV3Ќ
activation_13/ReluRelu+batch_normalization_10/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:          2
activation_13/ReluЮ
	add_4/addAddV2 activation_12/Relu:activations:0 activation_13/Relu:activations:0*
T0*/
_output_shapes
:          2
	add_4/add│
conv2d_12/Conv2D/ReadVariableOpReadVariableOp(conv2d_12_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02!
conv2d_12/Conv2D/ReadVariableOp╚
conv2d_12/Conv2DConv2Dadd_4/add:z:0'conv2d_12/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:          *
paddingSAME*
strides
2
conv2d_12/Conv2Dф
 conv2d_12/BiasAdd/ReadVariableOpReadVariableOp)conv2d_12_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_12/BiasAdd/ReadVariableOp░
conv2d_12/BiasAddBiasAddconv2d_12/Conv2D:output:0(conv2d_12/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          2
conv2d_12/BiasAdd╣
%batch_normalization_11/ReadVariableOpReadVariableOp.batch_normalization_11_readvariableop_resource*
_output_shapes
: *
dtype02'
%batch_normalization_11/ReadVariableOp┐
'batch_normalization_11/ReadVariableOp_1ReadVariableOp0batch_normalization_11_readvariableop_1_resource*
_output_shapes
: *
dtype02)
'batch_normalization_11/ReadVariableOp_1В
6batch_normalization_11/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_11_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype028
6batch_normalization_11/FusedBatchNormV3/ReadVariableOpЫ
8batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_11_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02:
8batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1У
'batch_normalization_11/FusedBatchNormV3FusedBatchNormV3conv2d_12/BiasAdd:output:0-batch_normalization_11/ReadVariableOp:value:0/batch_normalization_11/ReadVariableOp_1:value:0>batch_normalization_11/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:          : : : : :*
epsilon%oЃ:*
is_training( 2)
'batch_normalization_11/FusedBatchNormV3Ќ
activation_14/ReluRelu+batch_normalization_11/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:          2
activation_14/Reluі
	add_5/addAddV2add_4/add:z:0 activation_14/Relu:activations:0*
T0*/
_output_shapes
:          2
	add_5/addy
activation_15/ReluReluadd_5/add:z:0*
T0*/
_output_shapes
:          2
activation_15/Relu~
up_sampling2d_2/ShapeShape activation_15/Relu:activations:0*
T0*
_output_shapes
:2
up_sampling2d_2/Shapeћ
#up_sampling2d_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2%
#up_sampling2d_2/strided_slice/stackў
%up_sampling2d_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%up_sampling2d_2/strided_slice/stack_1ў
%up_sampling2d_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%up_sampling2d_2/strided_slice/stack_2«
up_sampling2d_2/strided_sliceStridedSliceup_sampling2d_2/Shape:output:0,up_sampling2d_2/strided_slice/stack:output:0.up_sampling2d_2/strided_slice/stack_1:output:0.up_sampling2d_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
up_sampling2d_2/strided_slice
up_sampling2d_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d_2/Constъ
up_sampling2d_2/mulMul&up_sampling2d_2/strided_slice:output:0up_sampling2d_2/Const:output:0*
T0*
_output_shapes
:2
up_sampling2d_2/mulё
,up_sampling2d_2/resize/ResizeNearestNeighborResizeNearestNeighbor activation_15/Relu:activations:0up_sampling2d_2/mul:z:0*
T0*/
_output_shapes
:          *
half_pixel_centers(2.
,up_sampling2d_2/resize/ResizeNearestNeighbor│
conv2d_13/Conv2D/ReadVariableOpReadVariableOp(conv2d_13_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02!
conv2d_13/Conv2D/ReadVariableOpЭ
conv2d_13/Conv2DConv2D=up_sampling2d_2/resize/ResizeNearestNeighbor:resized_images:0'conv2d_13/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:          *
paddingSAME*
strides
2
conv2d_13/Conv2Dф
 conv2d_13/BiasAdd/ReadVariableOpReadVariableOp)conv2d_13_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_13/BiasAdd/ReadVariableOp░
conv2d_13/BiasAddBiasAddconv2d_13/Conv2D:output:0(conv2d_13/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          2
conv2d_13/BiasAdd╣
%batch_normalization_12/ReadVariableOpReadVariableOp.batch_normalization_12_readvariableop_resource*
_output_shapes
: *
dtype02'
%batch_normalization_12/ReadVariableOp┐
'batch_normalization_12/ReadVariableOp_1ReadVariableOp0batch_normalization_12_readvariableop_1_resource*
_output_shapes
: *
dtype02)
'batch_normalization_12/ReadVariableOp_1В
6batch_normalization_12/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_12_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype028
6batch_normalization_12/FusedBatchNormV3/ReadVariableOpЫ
8batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_12_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02:
8batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1У
'batch_normalization_12/FusedBatchNormV3FusedBatchNormV3conv2d_13/BiasAdd:output:0-batch_normalization_12/ReadVariableOp:value:0/batch_normalization_12/ReadVariableOp_1:value:0>batch_normalization_12/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:          : : : : :*
epsilon%oЃ:*
is_training( 2)
'batch_normalization_12/FusedBatchNormV3Ќ
activation_16/ReluRelu+batch_normalization_12/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:          2
activation_16/Relu│
conv2d_14/Conv2D/ReadVariableOpReadVariableOp(conv2d_14_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02!
conv2d_14/Conv2D/ReadVariableOp█
conv2d_14/Conv2DConv2D activation_16/Relu:activations:0'conv2d_14/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:          *
paddingSAME*
strides
2
conv2d_14/Conv2Dф
 conv2d_14/BiasAdd/ReadVariableOpReadVariableOp)conv2d_14_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_14/BiasAdd/ReadVariableOp░
conv2d_14/BiasAddBiasAddconv2d_14/Conv2D:output:0(conv2d_14/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          2
conv2d_14/BiasAdd╣
%batch_normalization_13/ReadVariableOpReadVariableOp.batch_normalization_13_readvariableop_resource*
_output_shapes
: *
dtype02'
%batch_normalization_13/ReadVariableOp┐
'batch_normalization_13/ReadVariableOp_1ReadVariableOp0batch_normalization_13_readvariableop_1_resource*
_output_shapes
: *
dtype02)
'batch_normalization_13/ReadVariableOp_1В
6batch_normalization_13/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_13_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype028
6batch_normalization_13/FusedBatchNormV3/ReadVariableOpЫ
8batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_13_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02:
8batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1У
'batch_normalization_13/FusedBatchNormV3FusedBatchNormV3conv2d_14/BiasAdd:output:0-batch_normalization_13/ReadVariableOp:value:0/batch_normalization_13/ReadVariableOp_1:value:0>batch_normalization_13/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:          : : : : :*
epsilon%oЃ:*
is_training( 2)
'batch_normalization_13/FusedBatchNormV3Ќ
activation_17/ReluRelu+batch_normalization_13/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:          2
activation_17/ReluЮ
	add_6/addAddV2 activation_16/Relu:activations:0 activation_17/Relu:activations:0*
T0*/
_output_shapes
:          2
	add_6/add│
conv2d_15/Conv2D/ReadVariableOpReadVariableOp(conv2d_15_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02!
conv2d_15/Conv2D/ReadVariableOp╚
conv2d_15/Conv2DConv2Dadd_6/add:z:0'conv2d_15/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:          *
paddingSAME*
strides
2
conv2d_15/Conv2Dф
 conv2d_15/BiasAdd/ReadVariableOpReadVariableOp)conv2d_15_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_15/BiasAdd/ReadVariableOp░
conv2d_15/BiasAddBiasAddconv2d_15/Conv2D:output:0(conv2d_15/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          2
conv2d_15/BiasAdd╣
%batch_normalization_14/ReadVariableOpReadVariableOp.batch_normalization_14_readvariableop_resource*
_output_shapes
: *
dtype02'
%batch_normalization_14/ReadVariableOp┐
'batch_normalization_14/ReadVariableOp_1ReadVariableOp0batch_normalization_14_readvariableop_1_resource*
_output_shapes
: *
dtype02)
'batch_normalization_14/ReadVariableOp_1В
6batch_normalization_14/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_14_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype028
6batch_normalization_14/FusedBatchNormV3/ReadVariableOpЫ
8batch_normalization_14/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_14_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02:
8batch_normalization_14/FusedBatchNormV3/ReadVariableOp_1У
'batch_normalization_14/FusedBatchNormV3FusedBatchNormV3conv2d_15/BiasAdd:output:0-batch_normalization_14/ReadVariableOp:value:0/batch_normalization_14/ReadVariableOp_1:value:0>batch_normalization_14/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_14/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:          : : : : :*
epsilon%oЃ:*
is_training( 2)
'batch_normalization_14/FusedBatchNormV3Ќ
activation_18/ReluRelu+batch_normalization_14/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:          2
activation_18/Reluі
	add_7/addAddV2add_6/add:z:0 activation_18/Relu:activations:0*
T0*/
_output_shapes
:          2
	add_7/addy
activation_19/ReluReluadd_7/add:z:0*
T0*/
_output_shapes
:          2
activation_19/Relu~
up_sampling2d_3/ShapeShape activation_19/Relu:activations:0*
T0*
_output_shapes
:2
up_sampling2d_3/Shapeћ
#up_sampling2d_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2%
#up_sampling2d_3/strided_slice/stackў
%up_sampling2d_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%up_sampling2d_3/strided_slice/stack_1ў
%up_sampling2d_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%up_sampling2d_3/strided_slice/stack_2«
up_sampling2d_3/strided_sliceStridedSliceup_sampling2d_3/Shape:output:0,up_sampling2d_3/strided_slice/stack:output:0.up_sampling2d_3/strided_slice/stack_1:output:0.up_sampling2d_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
up_sampling2d_3/strided_slice
up_sampling2d_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d_3/Constъ
up_sampling2d_3/mulMul&up_sampling2d_3/strided_slice:output:0up_sampling2d_3/Const:output:0*
T0*
_output_shapes
:2
up_sampling2d_3/mulё
,up_sampling2d_3/resize/ResizeNearestNeighborResizeNearestNeighbor activation_19/Relu:activations:0up_sampling2d_3/mul:z:0*
T0*/
_output_shapes
:          *
half_pixel_centers(2.
,up_sampling2d_3/resize/ResizeNearestNeighbor│
conv2d_16/Conv2D/ReadVariableOpReadVariableOp(conv2d_16_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02!
conv2d_16/Conv2D/ReadVariableOpЭ
conv2d_16/Conv2DConv2D=up_sampling2d_3/resize/ResizeNearestNeighbor:resized_images:0'conv2d_16/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:          *
paddingSAME*
strides
2
conv2d_16/Conv2Dф
 conv2d_16/BiasAdd/ReadVariableOpReadVariableOp)conv2d_16_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_16/BiasAdd/ReadVariableOp░
conv2d_16/BiasAddBiasAddconv2d_16/Conv2D:output:0(conv2d_16/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          2
conv2d_16/BiasAdd╣
%batch_normalization_15/ReadVariableOpReadVariableOp.batch_normalization_15_readvariableop_resource*
_output_shapes
: *
dtype02'
%batch_normalization_15/ReadVariableOp┐
'batch_normalization_15/ReadVariableOp_1ReadVariableOp0batch_normalization_15_readvariableop_1_resource*
_output_shapes
: *
dtype02)
'batch_normalization_15/ReadVariableOp_1В
6batch_normalization_15/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_15_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype028
6batch_normalization_15/FusedBatchNormV3/ReadVariableOpЫ
8batch_normalization_15/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_15_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02:
8batch_normalization_15/FusedBatchNormV3/ReadVariableOp_1У
'batch_normalization_15/FusedBatchNormV3FusedBatchNormV3conv2d_16/BiasAdd:output:0-batch_normalization_15/ReadVariableOp:value:0/batch_normalization_15/ReadVariableOp_1:value:0>batch_normalization_15/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_15/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:          : : : : :*
epsilon%oЃ:*
is_training( 2)
'batch_normalization_15/FusedBatchNormV3Ќ
activation_20/ReluRelu+batch_normalization_15/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:          2
activation_20/Relu│
conv2d_17/Conv2D/ReadVariableOpReadVariableOp(conv2d_17_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_17/Conv2D/ReadVariableOp█
conv2d_17/Conv2DConv2D activation_20/Relu:activations:0'conv2d_17/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
paddingSAME*
strides
2
conv2d_17/Conv2Dф
 conv2d_17/BiasAdd/ReadVariableOpReadVariableOp)conv2d_17_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2d_17/BiasAdd/ReadVariableOp░
conv2d_17/BiasAddBiasAddconv2d_17/Conv2D:output:0(conv2d_17/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         2
conv2d_17/BiasAdd╣
%batch_normalization_16/ReadVariableOpReadVariableOp.batch_normalization_16_readvariableop_resource*
_output_shapes
:*
dtype02'
%batch_normalization_16/ReadVariableOp┐
'batch_normalization_16/ReadVariableOp_1ReadVariableOp0batch_normalization_16_readvariableop_1_resource*
_output_shapes
:*
dtype02)
'batch_normalization_16/ReadVariableOp_1В
6batch_normalization_16/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_16_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype028
6batch_normalization_16/FusedBatchNormV3/ReadVariableOpЫ
8batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_16_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02:
8batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1У
'batch_normalization_16/FusedBatchNormV3FusedBatchNormV3conv2d_17/BiasAdd:output:0-batch_normalization_16/ReadVariableOp:value:0/batch_normalization_16/ReadVariableOp_1:value:0>batch_normalization_16/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         :::::*
epsilon%oЃ:*
is_training( 2)
'batch_normalization_16/FusedBatchNormV3Ќ
activation_21/ReluRelu+batch_normalization_16/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:         2
activation_21/Relu|
IdentityIdentity activation_21/Relu:activations:0*
T0*/
_output_shapes
:         2

Identity"
identityIdentity:output:0*­
_input_shapesя
█:         :::::::::::::::::::::::::::::::::::::::::::::::::W S
/
_output_shapes
:         
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
: :!

_output_shapes
: :"

_output_shapes
: :#

_output_shapes
: :$

_output_shapes
: :%

_output_shapes
: :&

_output_shapes
: :'

_output_shapes
: :(

_output_shapes
: :)

_output_shapes
: :*

_output_shapes
: :+

_output_shapes
: :,

_output_shapes
: :-

_output_shapes
: :.

_output_shapes
: :/

_output_shapes
: :0

_output_shapes
: 
њ
І
R__inference_batch_normalization_15_layer_call_and_return_conditional_losses_254252

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityѕt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOpГ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1▄
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                            : : : : :*
epsilon%oЃ:*
is_training( 2
FusedBatchNormV3ѓ
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+                            2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                            :::::i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
У$
┘
R__inference_batch_normalization_11_layer_call_and_return_conditional_losses_257040

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityѕб#AssignMovingAvg/AssignSubVariableOpб%AssignMovingAvg_1/AssignSubVariableOpt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOpГ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1╔
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                            : : : : :*
epsilon%oЃ:2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
Const░
AssignMovingAvg/sub/xConst*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  ђ?2
AssignMovingAvg/sub/x┐
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const:output:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/subЦ
AssignMovingAvg/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOpя
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/sub_1К
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/mulК
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp(fusedbatchnormv3_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpХ
AssignMovingAvg_1/sub/xConst*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  ђ?2
AssignMovingAvg_1/sub/xК
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const:output:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/subФ
 AssignMovingAvg_1/ReadVariableOpReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOpЖ
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/sub_1Л
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/mulН
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp*fusedbatchnormv3_readvariableop_1_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpл
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*A
_output_shapes/
-:+                            2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                            ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
а
e
I__inference_activation_18_layer_call_and_return_conditional_losses_254930

inputs
identityh
ReluReluinputs*
T0*A
_output_shapes/
-:+                            2
Reluђ
IdentityIdentityRelu:activations:0*
T0*A
_output_shapes/
-:+                            2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+                            :i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
В
J
.__inference_activation_18_layer_call_fn_257419

inputs
identity╚
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*A
_output_shapes/
-:+                            * 
_read_only_resource_inputs
 *3
config_proto#!

CPU

GPU2*0,1,2,3J 8*R
fMRK
I__inference_activation_18_layer_call_and_return_conditional_losses_2549302
PartitionedCallє
IdentityIdentityPartitionedCall:output:0*
T0*A
_output_shapes/
-:+                            2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+                            :i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
╔
І
R__inference_batch_normalization_11_layer_call_and_return_conditional_losses_254686

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityѕt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOpГ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1╩
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:          : : : : :*
epsilon%oЃ:*
is_training( 2
FusedBatchNormV3p
IdentityIdentityFusedBatchNormV3:y:0*
T0*/
_output_shapes
:          2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:          :::::W S
/
_output_shapes
:          
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
ж

*__inference_conv2d_12_layer_call_fn_253511

inputs
unknown
	unknown_0
identityѕбStatefulPartitionedCallШ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*A
_output_shapes/
-:+                            *$
_read_only_resource_inputs
*3
config_proto#!

CPU

GPU2*0,1,2,3J 8*N
fIRG
E__inference_conv2d_12_layer_call_and_return_conditional_losses_2535012
StatefulPartitionedCallе
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                            2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+                            ::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
В
J
.__inference_activation_16_layer_call_fn_257213

inputs
identity╚
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*A
_output_shapes/
-:+                            * 
_read_only_resource_inputs
 *3
config_proto#!

CPU

GPU2*0,1,2,3J 8*R
fMRK
I__inference_activation_16_layer_call_and_return_conditional_losses_2548092
PartitionedCallє
IdentityIdentityPartitionedCall:output:0*
T0*A
_output_shapes/
-:+                            2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+                            :i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
 л
»
"__inference__traced_restore_257962
file_prefix%
!assignvariableop_conv2d_10_kernel%
!assignvariableop_1_conv2d_10_bias2
.assignvariableop_2_batch_normalization_9_gamma1
-assignvariableop_3_batch_normalization_9_beta8
4assignvariableop_4_batch_normalization_9_moving_mean<
8assignvariableop_5_batch_normalization_9_moving_variance'
#assignvariableop_6_conv2d_11_kernel%
!assignvariableop_7_conv2d_11_bias3
/assignvariableop_8_batch_normalization_10_gamma2
.assignvariableop_9_batch_normalization_10_beta:
6assignvariableop_10_batch_normalization_10_moving_mean>
:assignvariableop_11_batch_normalization_10_moving_variance(
$assignvariableop_12_conv2d_12_kernel&
"assignvariableop_13_conv2d_12_bias4
0assignvariableop_14_batch_normalization_11_gamma3
/assignvariableop_15_batch_normalization_11_beta:
6assignvariableop_16_batch_normalization_11_moving_mean>
:assignvariableop_17_batch_normalization_11_moving_variance(
$assignvariableop_18_conv2d_13_kernel&
"assignvariableop_19_conv2d_13_bias4
0assignvariableop_20_batch_normalization_12_gamma3
/assignvariableop_21_batch_normalization_12_beta:
6assignvariableop_22_batch_normalization_12_moving_mean>
:assignvariableop_23_batch_normalization_12_moving_variance(
$assignvariableop_24_conv2d_14_kernel&
"assignvariableop_25_conv2d_14_bias4
0assignvariableop_26_batch_normalization_13_gamma3
/assignvariableop_27_batch_normalization_13_beta:
6assignvariableop_28_batch_normalization_13_moving_mean>
:assignvariableop_29_batch_normalization_13_moving_variance(
$assignvariableop_30_conv2d_15_kernel&
"assignvariableop_31_conv2d_15_bias4
0assignvariableop_32_batch_normalization_14_gamma3
/assignvariableop_33_batch_normalization_14_beta:
6assignvariableop_34_batch_normalization_14_moving_mean>
:assignvariableop_35_batch_normalization_14_moving_variance(
$assignvariableop_36_conv2d_16_kernel&
"assignvariableop_37_conv2d_16_bias4
0assignvariableop_38_batch_normalization_15_gamma3
/assignvariableop_39_batch_normalization_15_beta:
6assignvariableop_40_batch_normalization_15_moving_mean>
:assignvariableop_41_batch_normalization_15_moving_variance(
$assignvariableop_42_conv2d_17_kernel&
"assignvariableop_43_conv2d_17_bias4
0assignvariableop_44_batch_normalization_16_gamma3
/assignvariableop_45_batch_normalization_16_beta:
6assignvariableop_46_batch_normalization_16_moving_mean>
:assignvariableop_47_batch_normalization_16_moving_variance
identity_49ѕбAssignVariableOpбAssignVariableOp_1бAssignVariableOp_10бAssignVariableOp_11бAssignVariableOp_12бAssignVariableOp_13бAssignVariableOp_14бAssignVariableOp_15бAssignVariableOp_16бAssignVariableOp_17бAssignVariableOp_18бAssignVariableOp_19бAssignVariableOp_2бAssignVariableOp_20бAssignVariableOp_21бAssignVariableOp_22бAssignVariableOp_23бAssignVariableOp_24бAssignVariableOp_25бAssignVariableOp_26бAssignVariableOp_27бAssignVariableOp_28бAssignVariableOp_29бAssignVariableOp_3бAssignVariableOp_30бAssignVariableOp_31бAssignVariableOp_32бAssignVariableOp_33бAssignVariableOp_34бAssignVariableOp_35бAssignVariableOp_36бAssignVariableOp_37бAssignVariableOp_38бAssignVariableOp_39бAssignVariableOp_4бAssignVariableOp_40бAssignVariableOp_41бAssignVariableOp_42бAssignVariableOp_43бAssignVariableOp_44бAssignVariableOp_45бAssignVariableOp_46бAssignVariableOp_47бAssignVariableOp_5бAssignVariableOp_6бAssignVariableOp_7бAssignVariableOp_8бAssignVariableOp_9б	RestoreV2бRestoreV2_1с
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:0*
dtype0*№
valueтBР0B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-9/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-9/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-9/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-11/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-11/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-11/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-13/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-13/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-13/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-15/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-15/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-15/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-15/moving_variance/.ATTRIBUTES/VARIABLE_VALUE2
RestoreV2/tensor_namesЬ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:0*
dtype0*s
valuejBh0B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slicesъ
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*о
_output_shapes├
└::::::::::::::::::::::::::::::::::::::::::::::::*>
dtypes4
2202
	RestoreV2X
IdentityIdentityRestoreV2:tensors:0*
T0*
_output_shapes
:2

IdentityЉ
AssignVariableOpAssignVariableOp!assignvariableop_conv2d_10_kernelIdentity:output:0*
_output_shapes
 *
dtype02
AssignVariableOp\

Identity_1IdentityRestoreV2:tensors:1*
T0*
_output_shapes
:2

Identity_1Ќ
AssignVariableOp_1AssignVariableOp!assignvariableop_1_conv2d_10_biasIdentity_1:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_1\

Identity_2IdentityRestoreV2:tensors:2*
T0*
_output_shapes
:2

Identity_2ц
AssignVariableOp_2AssignVariableOp.assignvariableop_2_batch_normalization_9_gammaIdentity_2:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_2\

Identity_3IdentityRestoreV2:tensors:3*
T0*
_output_shapes
:2

Identity_3Б
AssignVariableOp_3AssignVariableOp-assignvariableop_3_batch_normalization_9_betaIdentity_3:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_3\

Identity_4IdentityRestoreV2:tensors:4*
T0*
_output_shapes
:2

Identity_4ф
AssignVariableOp_4AssignVariableOp4assignvariableop_4_batch_normalization_9_moving_meanIdentity_4:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_4\

Identity_5IdentityRestoreV2:tensors:5*
T0*
_output_shapes
:2

Identity_5«
AssignVariableOp_5AssignVariableOp8assignvariableop_5_batch_normalization_9_moving_varianceIdentity_5:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_5\

Identity_6IdentityRestoreV2:tensors:6*
T0*
_output_shapes
:2

Identity_6Ў
AssignVariableOp_6AssignVariableOp#assignvariableop_6_conv2d_11_kernelIdentity_6:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_6\

Identity_7IdentityRestoreV2:tensors:7*
T0*
_output_shapes
:2

Identity_7Ќ
AssignVariableOp_7AssignVariableOp!assignvariableop_7_conv2d_11_biasIdentity_7:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_7\

Identity_8IdentityRestoreV2:tensors:8*
T0*
_output_shapes
:2

Identity_8Ц
AssignVariableOp_8AssignVariableOp/assignvariableop_8_batch_normalization_10_gammaIdentity_8:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_8\

Identity_9IdentityRestoreV2:tensors:9*
T0*
_output_shapes
:2

Identity_9ц
AssignVariableOp_9AssignVariableOp.assignvariableop_9_batch_normalization_10_betaIdentity_9:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_9_
Identity_10IdentityRestoreV2:tensors:10*
T0*
_output_shapes
:2
Identity_10»
AssignVariableOp_10AssignVariableOp6assignvariableop_10_batch_normalization_10_moving_meanIdentity_10:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_10_
Identity_11IdentityRestoreV2:tensors:11*
T0*
_output_shapes
:2
Identity_11│
AssignVariableOp_11AssignVariableOp:assignvariableop_11_batch_normalization_10_moving_varianceIdentity_11:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_11_
Identity_12IdentityRestoreV2:tensors:12*
T0*
_output_shapes
:2
Identity_12Ю
AssignVariableOp_12AssignVariableOp$assignvariableop_12_conv2d_12_kernelIdentity_12:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_12_
Identity_13IdentityRestoreV2:tensors:13*
T0*
_output_shapes
:2
Identity_13Џ
AssignVariableOp_13AssignVariableOp"assignvariableop_13_conv2d_12_biasIdentity_13:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_13_
Identity_14IdentityRestoreV2:tensors:14*
T0*
_output_shapes
:2
Identity_14Е
AssignVariableOp_14AssignVariableOp0assignvariableop_14_batch_normalization_11_gammaIdentity_14:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_14_
Identity_15IdentityRestoreV2:tensors:15*
T0*
_output_shapes
:2
Identity_15е
AssignVariableOp_15AssignVariableOp/assignvariableop_15_batch_normalization_11_betaIdentity_15:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_15_
Identity_16IdentityRestoreV2:tensors:16*
T0*
_output_shapes
:2
Identity_16»
AssignVariableOp_16AssignVariableOp6assignvariableop_16_batch_normalization_11_moving_meanIdentity_16:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_16_
Identity_17IdentityRestoreV2:tensors:17*
T0*
_output_shapes
:2
Identity_17│
AssignVariableOp_17AssignVariableOp:assignvariableop_17_batch_normalization_11_moving_varianceIdentity_17:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_17_
Identity_18IdentityRestoreV2:tensors:18*
T0*
_output_shapes
:2
Identity_18Ю
AssignVariableOp_18AssignVariableOp$assignvariableop_18_conv2d_13_kernelIdentity_18:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_18_
Identity_19IdentityRestoreV2:tensors:19*
T0*
_output_shapes
:2
Identity_19Џ
AssignVariableOp_19AssignVariableOp"assignvariableop_19_conv2d_13_biasIdentity_19:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_19_
Identity_20IdentityRestoreV2:tensors:20*
T0*
_output_shapes
:2
Identity_20Е
AssignVariableOp_20AssignVariableOp0assignvariableop_20_batch_normalization_12_gammaIdentity_20:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_20_
Identity_21IdentityRestoreV2:tensors:21*
T0*
_output_shapes
:2
Identity_21е
AssignVariableOp_21AssignVariableOp/assignvariableop_21_batch_normalization_12_betaIdentity_21:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_21_
Identity_22IdentityRestoreV2:tensors:22*
T0*
_output_shapes
:2
Identity_22»
AssignVariableOp_22AssignVariableOp6assignvariableop_22_batch_normalization_12_moving_meanIdentity_22:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_22_
Identity_23IdentityRestoreV2:tensors:23*
T0*
_output_shapes
:2
Identity_23│
AssignVariableOp_23AssignVariableOp:assignvariableop_23_batch_normalization_12_moving_varianceIdentity_23:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_23_
Identity_24IdentityRestoreV2:tensors:24*
T0*
_output_shapes
:2
Identity_24Ю
AssignVariableOp_24AssignVariableOp$assignvariableop_24_conv2d_14_kernelIdentity_24:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_24_
Identity_25IdentityRestoreV2:tensors:25*
T0*
_output_shapes
:2
Identity_25Џ
AssignVariableOp_25AssignVariableOp"assignvariableop_25_conv2d_14_biasIdentity_25:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_25_
Identity_26IdentityRestoreV2:tensors:26*
T0*
_output_shapes
:2
Identity_26Е
AssignVariableOp_26AssignVariableOp0assignvariableop_26_batch_normalization_13_gammaIdentity_26:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_26_
Identity_27IdentityRestoreV2:tensors:27*
T0*
_output_shapes
:2
Identity_27е
AssignVariableOp_27AssignVariableOp/assignvariableop_27_batch_normalization_13_betaIdentity_27:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_27_
Identity_28IdentityRestoreV2:tensors:28*
T0*
_output_shapes
:2
Identity_28»
AssignVariableOp_28AssignVariableOp6assignvariableop_28_batch_normalization_13_moving_meanIdentity_28:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_28_
Identity_29IdentityRestoreV2:tensors:29*
T0*
_output_shapes
:2
Identity_29│
AssignVariableOp_29AssignVariableOp:assignvariableop_29_batch_normalization_13_moving_varianceIdentity_29:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_29_
Identity_30IdentityRestoreV2:tensors:30*
T0*
_output_shapes
:2
Identity_30Ю
AssignVariableOp_30AssignVariableOp$assignvariableop_30_conv2d_15_kernelIdentity_30:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_30_
Identity_31IdentityRestoreV2:tensors:31*
T0*
_output_shapes
:2
Identity_31Џ
AssignVariableOp_31AssignVariableOp"assignvariableop_31_conv2d_15_biasIdentity_31:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_31_
Identity_32IdentityRestoreV2:tensors:32*
T0*
_output_shapes
:2
Identity_32Е
AssignVariableOp_32AssignVariableOp0assignvariableop_32_batch_normalization_14_gammaIdentity_32:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_32_
Identity_33IdentityRestoreV2:tensors:33*
T0*
_output_shapes
:2
Identity_33е
AssignVariableOp_33AssignVariableOp/assignvariableop_33_batch_normalization_14_betaIdentity_33:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_33_
Identity_34IdentityRestoreV2:tensors:34*
T0*
_output_shapes
:2
Identity_34»
AssignVariableOp_34AssignVariableOp6assignvariableop_34_batch_normalization_14_moving_meanIdentity_34:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_34_
Identity_35IdentityRestoreV2:tensors:35*
T0*
_output_shapes
:2
Identity_35│
AssignVariableOp_35AssignVariableOp:assignvariableop_35_batch_normalization_14_moving_varianceIdentity_35:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_35_
Identity_36IdentityRestoreV2:tensors:36*
T0*
_output_shapes
:2
Identity_36Ю
AssignVariableOp_36AssignVariableOp$assignvariableop_36_conv2d_16_kernelIdentity_36:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_36_
Identity_37IdentityRestoreV2:tensors:37*
T0*
_output_shapes
:2
Identity_37Џ
AssignVariableOp_37AssignVariableOp"assignvariableop_37_conv2d_16_biasIdentity_37:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_37_
Identity_38IdentityRestoreV2:tensors:38*
T0*
_output_shapes
:2
Identity_38Е
AssignVariableOp_38AssignVariableOp0assignvariableop_38_batch_normalization_15_gammaIdentity_38:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_38_
Identity_39IdentityRestoreV2:tensors:39*
T0*
_output_shapes
:2
Identity_39е
AssignVariableOp_39AssignVariableOp/assignvariableop_39_batch_normalization_15_betaIdentity_39:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_39_
Identity_40IdentityRestoreV2:tensors:40*
T0*
_output_shapes
:2
Identity_40»
AssignVariableOp_40AssignVariableOp6assignvariableop_40_batch_normalization_15_moving_meanIdentity_40:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_40_
Identity_41IdentityRestoreV2:tensors:41*
T0*
_output_shapes
:2
Identity_41│
AssignVariableOp_41AssignVariableOp:assignvariableop_41_batch_normalization_15_moving_varianceIdentity_41:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_41_
Identity_42IdentityRestoreV2:tensors:42*
T0*
_output_shapes
:2
Identity_42Ю
AssignVariableOp_42AssignVariableOp$assignvariableop_42_conv2d_17_kernelIdentity_42:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_42_
Identity_43IdentityRestoreV2:tensors:43*
T0*
_output_shapes
:2
Identity_43Џ
AssignVariableOp_43AssignVariableOp"assignvariableop_43_conv2d_17_biasIdentity_43:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_43_
Identity_44IdentityRestoreV2:tensors:44*
T0*
_output_shapes
:2
Identity_44Е
AssignVariableOp_44AssignVariableOp0assignvariableop_44_batch_normalization_16_gammaIdentity_44:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_44_
Identity_45IdentityRestoreV2:tensors:45*
T0*
_output_shapes
:2
Identity_45е
AssignVariableOp_45AssignVariableOp/assignvariableop_45_batch_normalization_16_betaIdentity_45:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_45_
Identity_46IdentityRestoreV2:tensors:46*
T0*
_output_shapes
:2
Identity_46»
AssignVariableOp_46AssignVariableOp6assignvariableop_46_batch_normalization_16_moving_meanIdentity_46:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_46_
Identity_47IdentityRestoreV2:tensors:47*
T0*
_output_shapes
:2
Identity_47│
AssignVariableOp_47AssignVariableOp:assignvariableop_47_batch_normalization_16_moving_varianceIdentity_47:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_47е
RestoreV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2_1/tensor_namesћ
RestoreV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
RestoreV2_1/shape_and_slices─
RestoreV2_1	RestoreV2file_prefix!RestoreV2_1/tensor_names:output:0%RestoreV2_1/shape_and_slices:output:0
^RestoreV2"/device:CPU:0*
_output_shapes
:*
dtypes
22
RestoreV2_19
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp■
Identity_48Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_48І	
Identity_49IdentityIdentity_48:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9
^RestoreV2^RestoreV2_1*
T0*
_output_shapes
: 2
Identity_49"#
identity_49Identity_49:output:0*О
_input_shapes┼
┬: ::::::::::::::::::::::::::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92
	RestoreV2	RestoreV22
RestoreV2_1RestoreV2_1:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
: :!

_output_shapes
: :"

_output_shapes
: :#

_output_shapes
: :$

_output_shapes
: :%

_output_shapes
: :&

_output_shapes
: :'

_output_shapes
: :(

_output_shapes
: :)

_output_shapes
: :*

_output_shapes
: :+

_output_shapes
: :,

_output_shapes
: :-

_output_shapes
: :.

_output_shapes
: :/

_output_shapes
: :0

_output_shapes
: 
╝	
Г
E__inference_conv2d_15_layer_call_and_return_conditional_losses_253961

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityѕЋ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02
Conv2D/ReadVariableOpх
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                            *
paddingSAME*
strides
2
Conv2Dї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOpџ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                            2	
BiasAdd~
IdentityIdentityBiasAdd:output:0*
T0*A
_output_shapes/
-:+                            2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+                            :::i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
Ћ
R
&__inference_add_7_layer_call_fn_257431
inputs_0
inputs_1
identity═
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*A
_output_shapes/
-:+                            * 
_read_only_resource_inputs
 *3
config_proto#!

CPU

GPU2*0,1,2,3J 8*J
fERC
A__inference_add_7_layer_call_and_return_conditional_losses_2549442
PartitionedCallє
IdentityIdentityPartitionedCall:output:0*
T0*A
_output_shapes/
-:+                            2

Identity"
identityIdentity:output:0*m
_input_shapes\
Z:+                            :+                            :k g
A
_output_shapes/
-:+                            
"
_user_specified_name
inputs/0:kg
A
_output_shapes/
-:+                            
"
_user_specified_name
inputs/1
д
Н
(__inference_model_2_layer_call_fn_255440
input_3
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34

unknown_35

unknown_36

unknown_37

unknown_38

unknown_39

unknown_40

unknown_41

unknown_42

unknown_43

unknown_44

unknown_45

unknown_46
identityѕбStatefulPartitionedCallЯ
StatefulPartitionedCallStatefulPartitionedCallinput_3unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46*<
Tin5
321*
Tout
2*A
_output_shapes/
-:+                           *B
_read_only_resource_inputs$
" 	
 !"%&'(+,-.*3
config_proto#!

CPU

GPU2*0,1,2,3J 8*L
fGRE
C__inference_model_2_layer_call_and_return_conditional_losses_2553412
StatefulPartitionedCallе
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*­
_input_shapesя
█:         ::::::::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:         
!
_user_specified_name	input_3:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
: :!

_output_shapes
: :"

_output_shapes
: :#

_output_shapes
: :$

_output_shapes
: :%

_output_shapes
: :&

_output_shapes
: :'

_output_shapes
: :(

_output_shapes
: :)

_output_shapes
: :*

_output_shapes
: :+

_output_shapes
: :,

_output_shapes
: :-

_output_shapes
: :.

_output_shapes
: :/

_output_shapes
: :0

_output_shapes
: 
■
ф
7__inference_batch_normalization_10_layer_call_fn_256887

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityѕбStatefulPartitionedCallЏ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*A
_output_shapes/
-:+                            *$
_read_only_resource_inputs
*3
config_proto#!

CPU

GPU2*0,1,2,3J 8*[
fVRT
R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_2534482
StatefulPartitionedCallе
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                            2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                            ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
ѕ
g
K__inference_up_sampling2d_3_layer_call_and_return_conditional_losses_254110

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2╬
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
strided_slice_
ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
Const^
mulMulstrided_slice:output:0Const:output:0*
T0*
_output_shapes
:2
mulН
resize/ResizeNearestNeighborResizeNearestNeighborinputsmul:z:0*
T0*J
_output_shapes8
6:4                                    *
half_pixel_centers(2
resize/ResizeNearestNeighborц
IdentityIdentity-resize/ResizeNearestNeighbor:resized_images:0*
T0*J
_output_shapes8
6:4                                    2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
У$
┘
R__inference_batch_normalization_11_layer_call_and_return_conditional_losses_253595

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityѕб#AssignMovingAvg/AssignSubVariableOpб%AssignMovingAvg_1/AssignSubVariableOpt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOpГ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1╔
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                            : : : : :*
epsilon%oЃ:2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
Const░
AssignMovingAvg/sub/xConst*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  ђ?2
AssignMovingAvg/sub/x┐
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const:output:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/subЦ
AssignMovingAvg/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOpя
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/sub_1К
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/mulК
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp(fusedbatchnormv3_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpХ
AssignMovingAvg_1/sub/xConst*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  ђ?2
AssignMovingAvg_1/sub/xК
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const:output:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/subФ
 AssignMovingAvg_1/ReadVariableOpReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOpЖ
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/sub_1Л
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/mulН
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp*fusedbatchnormv3_readvariableop_1_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpл
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*A
_output_shapes/
-:+                            2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                            ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
е
R
&__inference_add_5_layer_call_fn_257106
inputs_0
inputs_1
identity╗
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*/
_output_shapes
:          * 
_read_only_resource_inputs
 *3
config_proto#!

CPU

GPU2*0,1,2,3J 8*J
fERC
A__inference_add_5_layer_call_and_return_conditional_losses_2547412
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:          2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:          :          :Y U
/
_output_shapes
:          
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:          
"
_user_specified_name
inputs/1
В
J
.__inference_activation_21_layer_call_fn_257635

inputs
identity╚
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*A
_output_shapes/
-:+                           * 
_read_only_resource_inputs
 *3
config_proto#!

CPU

GPU2*0,1,2,3J 8*R
fMRK
I__inference_activation_21_layer_call_and_return_conditional_losses_2550652
PartitionedCallє
IdentityIdentityPartitionedCall:output:0*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+                           :i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
■
ф
7__inference_batch_normalization_15_layer_call_fn_257515

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityѕбStatefulPartitionedCallЏ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*A
_output_shapes/
-:+                            *$
_read_only_resource_inputs
*3
config_proto#!

CPU

GPU2*0,1,2,3J 8*[
fVRT
R__inference_batch_normalization_15_layer_call_and_return_conditional_losses_2542212
StatefulPartitionedCallе
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                            2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                            ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Цl
ќ
__inference__traced_save_257806
file_prefix/
+savev2_conv2d_10_kernel_read_readvariableop-
)savev2_conv2d_10_bias_read_readvariableop:
6savev2_batch_normalization_9_gamma_read_readvariableop9
5savev2_batch_normalization_9_beta_read_readvariableop@
<savev2_batch_normalization_9_moving_mean_read_readvariableopD
@savev2_batch_normalization_9_moving_variance_read_readvariableop/
+savev2_conv2d_11_kernel_read_readvariableop-
)savev2_conv2d_11_bias_read_readvariableop;
7savev2_batch_normalization_10_gamma_read_readvariableop:
6savev2_batch_normalization_10_beta_read_readvariableopA
=savev2_batch_normalization_10_moving_mean_read_readvariableopE
Asavev2_batch_normalization_10_moving_variance_read_readvariableop/
+savev2_conv2d_12_kernel_read_readvariableop-
)savev2_conv2d_12_bias_read_readvariableop;
7savev2_batch_normalization_11_gamma_read_readvariableop:
6savev2_batch_normalization_11_beta_read_readvariableopA
=savev2_batch_normalization_11_moving_mean_read_readvariableopE
Asavev2_batch_normalization_11_moving_variance_read_readvariableop/
+savev2_conv2d_13_kernel_read_readvariableop-
)savev2_conv2d_13_bias_read_readvariableop;
7savev2_batch_normalization_12_gamma_read_readvariableop:
6savev2_batch_normalization_12_beta_read_readvariableopA
=savev2_batch_normalization_12_moving_mean_read_readvariableopE
Asavev2_batch_normalization_12_moving_variance_read_readvariableop/
+savev2_conv2d_14_kernel_read_readvariableop-
)savev2_conv2d_14_bias_read_readvariableop;
7savev2_batch_normalization_13_gamma_read_readvariableop:
6savev2_batch_normalization_13_beta_read_readvariableopA
=savev2_batch_normalization_13_moving_mean_read_readvariableopE
Asavev2_batch_normalization_13_moving_variance_read_readvariableop/
+savev2_conv2d_15_kernel_read_readvariableop-
)savev2_conv2d_15_bias_read_readvariableop;
7savev2_batch_normalization_14_gamma_read_readvariableop:
6savev2_batch_normalization_14_beta_read_readvariableopA
=savev2_batch_normalization_14_moving_mean_read_readvariableopE
Asavev2_batch_normalization_14_moving_variance_read_readvariableop/
+savev2_conv2d_16_kernel_read_readvariableop-
)savev2_conv2d_16_bias_read_readvariableop;
7savev2_batch_normalization_15_gamma_read_readvariableop:
6savev2_batch_normalization_15_beta_read_readvariableopA
=savev2_batch_normalization_15_moving_mean_read_readvariableopE
Asavev2_batch_normalization_15_moving_variance_read_readvariableop/
+savev2_conv2d_17_kernel_read_readvariableop-
)savev2_conv2d_17_bias_read_readvariableop;
7savev2_batch_normalization_16_gamma_read_readvariableop:
6savev2_batch_normalization_16_beta_read_readvariableopA
=savev2_batch_normalization_16_moving_mean_read_readvariableopE
Asavev2_batch_normalization_16_moving_variance_read_readvariableop
savev2_1_const

identity_1ѕбMergeV2CheckpointsбSaveV2бSaveV2_1Ј
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
ConstЇ
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_a13ca7b3ea2c439db28aa24660d8a632/part2	
Const_1І
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shardд
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilenameП
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:0*
dtype0*№
valueтBР0B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-9/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-9/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-9/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-11/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-11/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-11/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-13/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-13/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-13/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-15/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-15/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-15/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-15/moving_variance/.ATTRIBUTES/VARIABLE_VALUE2
SaveV2/tensor_namesУ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:0*
dtype0*s
valuejBh0B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices»
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0+savev2_conv2d_10_kernel_read_readvariableop)savev2_conv2d_10_bias_read_readvariableop6savev2_batch_normalization_9_gamma_read_readvariableop5savev2_batch_normalization_9_beta_read_readvariableop<savev2_batch_normalization_9_moving_mean_read_readvariableop@savev2_batch_normalization_9_moving_variance_read_readvariableop+savev2_conv2d_11_kernel_read_readvariableop)savev2_conv2d_11_bias_read_readvariableop7savev2_batch_normalization_10_gamma_read_readvariableop6savev2_batch_normalization_10_beta_read_readvariableop=savev2_batch_normalization_10_moving_mean_read_readvariableopAsavev2_batch_normalization_10_moving_variance_read_readvariableop+savev2_conv2d_12_kernel_read_readvariableop)savev2_conv2d_12_bias_read_readvariableop7savev2_batch_normalization_11_gamma_read_readvariableop6savev2_batch_normalization_11_beta_read_readvariableop=savev2_batch_normalization_11_moving_mean_read_readvariableopAsavev2_batch_normalization_11_moving_variance_read_readvariableop+savev2_conv2d_13_kernel_read_readvariableop)savev2_conv2d_13_bias_read_readvariableop7savev2_batch_normalization_12_gamma_read_readvariableop6savev2_batch_normalization_12_beta_read_readvariableop=savev2_batch_normalization_12_moving_mean_read_readvariableopAsavev2_batch_normalization_12_moving_variance_read_readvariableop+savev2_conv2d_14_kernel_read_readvariableop)savev2_conv2d_14_bias_read_readvariableop7savev2_batch_normalization_13_gamma_read_readvariableop6savev2_batch_normalization_13_beta_read_readvariableop=savev2_batch_normalization_13_moving_mean_read_readvariableopAsavev2_batch_normalization_13_moving_variance_read_readvariableop+savev2_conv2d_15_kernel_read_readvariableop)savev2_conv2d_15_bias_read_readvariableop7savev2_batch_normalization_14_gamma_read_readvariableop6savev2_batch_normalization_14_beta_read_readvariableop=savev2_batch_normalization_14_moving_mean_read_readvariableopAsavev2_batch_normalization_14_moving_variance_read_readvariableop+savev2_conv2d_16_kernel_read_readvariableop)savev2_conv2d_16_bias_read_readvariableop7savev2_batch_normalization_15_gamma_read_readvariableop6savev2_batch_normalization_15_beta_read_readvariableop=savev2_batch_normalization_15_moving_mean_read_readvariableopAsavev2_batch_normalization_15_moving_variance_read_readvariableop+savev2_conv2d_17_kernel_read_readvariableop)savev2_conv2d_17_bias_read_readvariableop7savev2_batch_normalization_16_gamma_read_readvariableop6savev2_batch_normalization_16_beta_read_readvariableop=savev2_batch_normalization_16_moving_mean_read_readvariableopAsavev2_batch_normalization_16_moving_variance_read_readvariableop"/device:CPU:0*
_output_shapes
 *>
dtypes4
2202
SaveV2Ѓ
ShardedFilename_1/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B :2
ShardedFilename_1/shardг
ShardedFilename_1ShardedFilenameStringJoin:output:0 ShardedFilename_1/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename_1б
SaveV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2_1/tensor_namesј
SaveV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
SaveV2_1/shape_and_slices¤
SaveV2_1SaveV2ShardedFilename_1:filename:0SaveV2_1/tensor_names:output:0"SaveV2_1/shape_and_slices:output:0savev2_1_const^SaveV2"/device:CPU:0*
_output_shapes
 *
dtypes
22

SaveV2_1с
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0ShardedFilename_1:filename:0^SaveV2	^SaveV2_1"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixesг
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix	^SaveV2_1"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

IdentityЂ

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints^SaveV2	^SaveV2_1*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*Ў
_input_shapesЄ
ё: : : : : : : :  : : : : : :  : : : : : :  : : : : : :  : : : : : :  : : : : : :  : : : : : : :::::: 2(
MergeV2CheckpointsMergeV2Checkpoints2
SaveV2SaveV22
SaveV2_1SaveV2_1:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
:  : 

_output_shapes
: : 	

_output_shapes
: : 


_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
:  : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
:  : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
:  : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
:  :  

_output_shapes
: : !

_output_shapes
: : "

_output_shapes
: : #

_output_shapes
: : $

_output_shapes
: :,%(
&
_output_shapes
:  : &

_output_shapes
: : '

_output_shapes
: : (

_output_shapes
: : )

_output_shapes
: : *

_output_shapes
: :,+(
&
_output_shapes
: : ,

_output_shapes
:: -

_output_shapes
:: .

_output_shapes
:: /

_output_shapes
:: 0

_output_shapes
::1

_output_shapes
: 
Х
ф
7__inference_batch_normalization_10_layer_call_fn_256812

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityѕбStatefulPartitionedCallЅ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*/
_output_shapes
:          *$
_read_only_resource_inputs
*3
config_proto#!

CPU

GPU2*0,1,2,3J 8*[
fVRT
R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_2545512
StatefulPartitionedCallќ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:          2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:          ::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:          
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
У$
┘
R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_256856

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityѕб#AssignMovingAvg/AssignSubVariableOpб%AssignMovingAvg_1/AssignSubVariableOpt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOpГ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1╔
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                            : : : : :*
epsilon%oЃ:2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
Const░
AssignMovingAvg/sub/xConst*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  ђ?2
AssignMovingAvg/sub/x┐
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const:output:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/subЦ
AssignMovingAvg/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOpя
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/sub_1К
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/mulК
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp(fusedbatchnormv3_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpХ
AssignMovingAvg_1/sub/xConst*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  ђ?2
AssignMovingAvg_1/sub/xК
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const:output:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/subФ
 AssignMovingAvg_1/ReadVariableOpReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOpЖ
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/sub_1Л
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/mulН
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp*fusedbatchnormv3_readvariableop_1_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpл
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*A
_output_shapes/
-:+                            2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                            ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Б
J
.__inference_activation_15_layer_call_fn_257116

inputs
identityХ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*/
_output_shapes
:          * 
_read_only_resource_inputs
 *3
config_proto#!

CPU

GPU2*0,1,2,3J 8*R
fMRK
I__inference_activation_15_layer_call_and_return_conditional_losses_2547552
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:          2

Identity"
identityIdentity:output:0*.
_input_shapes
:          :W S
/
_output_shapes
:          
 
_user_specified_nameinputs
╗
k
A__inference_add_7_layer_call_and_return_conditional_losses_254944

inputs
inputs_1
identityq
addAddV2inputsinputs_1*
T0*A
_output_shapes/
-:+                            2
addu
IdentityIdentityadd:z:0*
T0*A
_output_shapes/
-:+                            2

Identity"
identityIdentity:output:0*m
_input_shapes\
Z:+                            :+                            :i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs:ie
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
╝	
Г
E__inference_conv2d_16_layer_call_and_return_conditional_losses_254127

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityѕЋ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02
Conv2D/ReadVariableOpх
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                            *
paddingSAME*
strides
2
Conv2Dї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOpџ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                            2	
BiasAdd~
IdentityIdentityBiasAdd:output:0*
T0*A
_output_shapes/
-:+                            2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+                            :::i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
¤
k
A__inference_add_5_layer_call_and_return_conditional_losses_254741

inputs
inputs_1
identity_
addAddV2inputsinputs_1*
T0*/
_output_shapes
:          2
addc
IdentityIdentityadd:z:0*
T0*/
_output_shapes
:          2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:          :          :W S
/
_output_shapes
:          
 
_user_specified_nameinputs:WS
/
_output_shapes
:          
 
_user_specified_nameinputs
њ
І
R__inference_batch_normalization_12_layer_call_and_return_conditional_losses_253792

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityѕt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOpГ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1▄
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                            : : : : :*
epsilon%oЃ:*
is_training( 2
FusedBatchNormV3ѓ
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+                            2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                            :::::i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
О
e
I__inference_activation_12_layer_call_and_return_conditional_losses_254508

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:          2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:          2

Identity"
identityIdentity:output:0*.
_input_shapes
:          :W S
/
_output_shapes
:          
 
_user_specified_nameinputs
ж

*__inference_conv2d_10_layer_call_fn_253217

inputs
unknown
	unknown_0
identityѕбStatefulPartitionedCallШ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*A
_output_shapes/
-:+                            *$
_read_only_resource_inputs
*3
config_proto#!

CPU

GPU2*0,1,2,3J 8*N
fIRG
E__inference_conv2d_10_layer_call_and_return_conditional_losses_2532072
StatefulPartitionedCallе
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                            2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+                           ::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
а
e
I__inference_activation_20_layer_call_and_return_conditional_losses_257533

inputs
identityh
ReluReluinputs*
T0*A
_output_shapes/
-:+                            2
Reluђ
IdentityIdentityRelu:activations:0*
T0*A
_output_shapes/
-:+                            2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+                            :i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
У$
┘
R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_253448

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityѕб#AssignMovingAvg/AssignSubVariableOpб%AssignMovingAvg_1/AssignSubVariableOpt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOpГ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1╔
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                            : : : : :*
epsilon%oЃ:2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
Const░
AssignMovingAvg/sub/xConst*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  ђ?2
AssignMovingAvg/sub/x┐
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const:output:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/subЦ
AssignMovingAvg/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOpя
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/sub_1К
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/mulК
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp(fusedbatchnormv3_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpХ
AssignMovingAvg_1/sub/xConst*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  ђ?2
AssignMovingAvg_1/sub/xК
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const:output:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/subФ
 AssignMovingAvg_1/ReadVariableOpReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOpЖ
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/sub_1Л
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/mulН
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp*fusedbatchnormv3_readvariableop_1_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpл
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*A
_output_shapes/
-:+                            2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                            ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Х
Н
(__inference_model_2_layer_call_fn_255673
input_3
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34

unknown_35

unknown_36

unknown_37

unknown_38

unknown_39

unknown_40

unknown_41

unknown_42

unknown_43

unknown_44

unknown_45

unknown_46
identityѕбStatefulPartitionedCall­
StatefulPartitionedCallStatefulPartitionedCallinput_3unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46*<
Tin5
321*
Tout
2*A
_output_shapes/
-:+                           *R
_read_only_resource_inputs4
20	
 !"#$%&'()*+,-./0*3
config_proto#!

CPU

GPU2*0,1,2,3J 8*L
fGRE
C__inference_model_2_layer_call_and_return_conditional_losses_2555742
StatefulPartitionedCallе
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*­
_input_shapesя
█:         ::::::::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:         
!
_user_specified_name	input_3:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
: :!

_output_shapes
: :"

_output_shapes
: :#

_output_shapes
: :$

_output_shapes
: :%

_output_shapes
: :&

_output_shapes
: :'

_output_shapes
: :(

_output_shapes
: :)

_output_shapes
: :*

_output_shapes
: :+

_output_shapes
: :,

_output_shapes
: :-

_output_shapes
: :.

_output_shapes
: :/

_output_shapes
: :0

_output_shapes
: 
Ћ
R
&__inference_add_6_layer_call_fn_257322
inputs_0
inputs_1
identity═
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*A
_output_shapes/
-:+                            * 
_read_only_resource_inputs
 *3
config_proto#!

CPU

GPU2*0,1,2,3J 8*J
fERC
A__inference_add_6_layer_call_and_return_conditional_losses_2548762
PartitionedCallє
IdentityIdentityPartitionedCall:output:0*
T0*A
_output_shapes/
-:+                            2

Identity"
identityIdentity:output:0*m
_input_shapes\
Z:+                            :+                            :k g
A
_output_shapes/
-:+                            
"
_user_specified_name
inputs/0:kg
A
_output_shapes/
-:+                            
"
_user_specified_name
inputs/1
Љ
і
Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_256627

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityѕt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOpГ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1▄
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                            : : : : :*
epsilon%oЃ:*
is_training( 2
FusedBatchNormV3ѓ
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+                            2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                            :::::i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Б
J
.__inference_activation_12_layer_call_fn_256738

inputs
identityХ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*/
_output_shapes
:          * 
_read_only_resource_inputs
 *3
config_proto#!

CPU

GPU2*0,1,2,3J 8*R
fMRK
I__inference_activation_12_layer_call_and_return_conditional_losses_2545082
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:          2

Identity"
identityIdentity:output:0*.
_input_shapes
:          :W S
/
_output_shapes
:          
 
_user_specified_nameinputs
а$
┘
R__inference_batch_normalization_11_layer_call_and_return_conditional_losses_254668

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityѕб#AssignMovingAvg/AssignSubVariableOpб%AssignMovingAvg_1/AssignSubVariableOpt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOpГ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1и
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:          : : : : :*
epsilon%oЃ:2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
Const░
AssignMovingAvg/sub/xConst*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  ђ?2
AssignMovingAvg/sub/x┐
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const:output:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/subЦ
AssignMovingAvg/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOpя
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/sub_1К
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/mulК
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp(fusedbatchnormv3_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpХ
AssignMovingAvg_1/sub/xConst*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  ђ?2
AssignMovingAvg_1/sub/xК
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const:output:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/subФ
 AssignMovingAvg_1/ReadVariableOpReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOpЖ
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/sub_1Л
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/mulН
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp*fusedbatchnormv3_readvariableop_1_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpЙ
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*/
_output_shapes
:          2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:          ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:W S
/
_output_shapes
:          
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
О
e
I__inference_activation_14_layer_call_and_return_conditional_losses_254727

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:          2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:          2

Identity"
identityIdentity:output:0*.
_input_shapes
:          :W S
/
_output_shapes
:          
 
_user_specified_nameinputs
а
e
I__inference_activation_19_layer_call_and_return_conditional_losses_257436

inputs
identityh
ReluReluinputs*
T0*A
_output_shapes/
-:+                            2
Reluђ
IdentityIdentityRelu:activations:0*
T0*A
_output_shapes/
-:+                            2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+                            :i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
ђ	
ф
7__inference_batch_normalization_11_layer_call_fn_257084

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityѕбStatefulPartitionedCallЮ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*A
_output_shapes/
-:+                            *&
_read_only_resource_inputs
*3
config_proto#!

CPU

GPU2*0,1,2,3J 8*[
fVRT
R__inference_batch_normalization_11_layer_call_and_return_conditional_losses_2536262
StatefulPartitionedCallе
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                            2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                            ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Х
ф
7__inference_batch_normalization_11_layer_call_fn_256996

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityѕбStatefulPartitionedCallЅ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*/
_output_shapes
:          *$
_read_only_resource_inputs
*3
config_proto#!

CPU

GPU2*0,1,2,3J 8*[
fVRT
R__inference_batch_normalization_11_layer_call_and_return_conditional_losses_2546682
StatefulPartitionedCallќ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:          2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:          ::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:          
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
а
e
I__inference_activation_21_layer_call_and_return_conditional_losses_257630

inputs
identityh
ReluReluinputs*
T0*A
_output_shapes/
-:+                           2
Reluђ
IdentityIdentityRelu:activations:0*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+                           :i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
О
e
I__inference_activation_14_layer_call_and_return_conditional_losses_257089

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:          2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:          2

Identity"
identityIdentity:output:0*.
_input_shapes
:          :W S
/
_output_shapes
:          
 
_user_specified_nameinputs
Ч
Е
6__inference_batch_normalization_9_layer_call_fn_256640

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityѕбStatefulPartitionedCallџ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*A
_output_shapes/
-:+                            *$
_read_only_resource_inputs
*3
config_proto#!

CPU

GPU2*0,1,2,3J 8*Z
fURS
Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_2533012
StatefulPartitionedCallе
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                            2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                            ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Єж
Н
!__inference__wrapped_model_253196
input_34
0model_2_conv2d_10_conv2d_readvariableop_resource5
1model_2_conv2d_10_biasadd_readvariableop_resource9
5model_2_batch_normalization_9_readvariableop_resource;
7model_2_batch_normalization_9_readvariableop_1_resourceJ
Fmodel_2_batch_normalization_9_fusedbatchnormv3_readvariableop_resourceL
Hmodel_2_batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource4
0model_2_conv2d_11_conv2d_readvariableop_resource5
1model_2_conv2d_11_biasadd_readvariableop_resource:
6model_2_batch_normalization_10_readvariableop_resource<
8model_2_batch_normalization_10_readvariableop_1_resourceK
Gmodel_2_batch_normalization_10_fusedbatchnormv3_readvariableop_resourceM
Imodel_2_batch_normalization_10_fusedbatchnormv3_readvariableop_1_resource4
0model_2_conv2d_12_conv2d_readvariableop_resource5
1model_2_conv2d_12_biasadd_readvariableop_resource:
6model_2_batch_normalization_11_readvariableop_resource<
8model_2_batch_normalization_11_readvariableop_1_resourceK
Gmodel_2_batch_normalization_11_fusedbatchnormv3_readvariableop_resourceM
Imodel_2_batch_normalization_11_fusedbatchnormv3_readvariableop_1_resource4
0model_2_conv2d_13_conv2d_readvariableop_resource5
1model_2_conv2d_13_biasadd_readvariableop_resource:
6model_2_batch_normalization_12_readvariableop_resource<
8model_2_batch_normalization_12_readvariableop_1_resourceK
Gmodel_2_batch_normalization_12_fusedbatchnormv3_readvariableop_resourceM
Imodel_2_batch_normalization_12_fusedbatchnormv3_readvariableop_1_resource4
0model_2_conv2d_14_conv2d_readvariableop_resource5
1model_2_conv2d_14_biasadd_readvariableop_resource:
6model_2_batch_normalization_13_readvariableop_resource<
8model_2_batch_normalization_13_readvariableop_1_resourceK
Gmodel_2_batch_normalization_13_fusedbatchnormv3_readvariableop_resourceM
Imodel_2_batch_normalization_13_fusedbatchnormv3_readvariableop_1_resource4
0model_2_conv2d_15_conv2d_readvariableop_resource5
1model_2_conv2d_15_biasadd_readvariableop_resource:
6model_2_batch_normalization_14_readvariableop_resource<
8model_2_batch_normalization_14_readvariableop_1_resourceK
Gmodel_2_batch_normalization_14_fusedbatchnormv3_readvariableop_resourceM
Imodel_2_batch_normalization_14_fusedbatchnormv3_readvariableop_1_resource4
0model_2_conv2d_16_conv2d_readvariableop_resource5
1model_2_conv2d_16_biasadd_readvariableop_resource:
6model_2_batch_normalization_15_readvariableop_resource<
8model_2_batch_normalization_15_readvariableop_1_resourceK
Gmodel_2_batch_normalization_15_fusedbatchnormv3_readvariableop_resourceM
Imodel_2_batch_normalization_15_fusedbatchnormv3_readvariableop_1_resource4
0model_2_conv2d_17_conv2d_readvariableop_resource5
1model_2_conv2d_17_biasadd_readvariableop_resource:
6model_2_batch_normalization_16_readvariableop_resource<
8model_2_batch_normalization_16_readvariableop_1_resourceK
Gmodel_2_batch_normalization_16_fusedbatchnormv3_readvariableop_resourceM
Imodel_2_batch_normalization_16_fusedbatchnormv3_readvariableop_1_resource
identityѕ╦
'model_2/conv2d_10/Conv2D/ReadVariableOpReadVariableOp0model_2_conv2d_10_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02)
'model_2/conv2d_10/Conv2D/ReadVariableOp┌
model_2/conv2d_10/Conv2DConv2Dinput_3/model_2/conv2d_10/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:          *
paddingSAME*
strides
2
model_2/conv2d_10/Conv2D┬
(model_2/conv2d_10/BiasAdd/ReadVariableOpReadVariableOp1model_2_conv2d_10_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02*
(model_2/conv2d_10/BiasAdd/ReadVariableOpл
model_2/conv2d_10/BiasAddBiasAdd!model_2/conv2d_10/Conv2D:output:00model_2/conv2d_10/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          2
model_2/conv2d_10/BiasAdd╬
,model_2/batch_normalization_9/ReadVariableOpReadVariableOp5model_2_batch_normalization_9_readvariableop_resource*
_output_shapes
: *
dtype02.
,model_2/batch_normalization_9/ReadVariableOpн
.model_2/batch_normalization_9/ReadVariableOp_1ReadVariableOp7model_2_batch_normalization_9_readvariableop_1_resource*
_output_shapes
: *
dtype020
.model_2/batch_normalization_9/ReadVariableOp_1Ђ
=model_2/batch_normalization_9/FusedBatchNormV3/ReadVariableOpReadVariableOpFmodel_2_batch_normalization_9_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02?
=model_2/batch_normalization_9/FusedBatchNormV3/ReadVariableOpЄ
?model_2/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpHmodel_2_batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02A
?model_2/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1џ
.model_2/batch_normalization_9/FusedBatchNormV3FusedBatchNormV3"model_2/conv2d_10/BiasAdd:output:04model_2/batch_normalization_9/ReadVariableOp:value:06model_2/batch_normalization_9/ReadVariableOp_1:value:0Emodel_2/batch_normalization_9/FusedBatchNormV3/ReadVariableOp:value:0Gmodel_2/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:          : : : : :*
epsilon%oЃ:*
is_training( 20
.model_2/batch_normalization_9/FusedBatchNormV3«
model_2/activation_12/ReluRelu2model_2/batch_normalization_9/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:          2
model_2/activation_12/Relu╦
'model_2/conv2d_11/Conv2D/ReadVariableOpReadVariableOp0model_2_conv2d_11_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02)
'model_2/conv2d_11/Conv2D/ReadVariableOpч
model_2/conv2d_11/Conv2DConv2D(model_2/activation_12/Relu:activations:0/model_2/conv2d_11/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:          *
paddingSAME*
strides
2
model_2/conv2d_11/Conv2D┬
(model_2/conv2d_11/BiasAdd/ReadVariableOpReadVariableOp1model_2_conv2d_11_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02*
(model_2/conv2d_11/BiasAdd/ReadVariableOpл
model_2/conv2d_11/BiasAddBiasAdd!model_2/conv2d_11/Conv2D:output:00model_2/conv2d_11/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          2
model_2/conv2d_11/BiasAddЛ
-model_2/batch_normalization_10/ReadVariableOpReadVariableOp6model_2_batch_normalization_10_readvariableop_resource*
_output_shapes
: *
dtype02/
-model_2/batch_normalization_10/ReadVariableOpО
/model_2/batch_normalization_10/ReadVariableOp_1ReadVariableOp8model_2_batch_normalization_10_readvariableop_1_resource*
_output_shapes
: *
dtype021
/model_2/batch_normalization_10/ReadVariableOp_1ё
>model_2/batch_normalization_10/FusedBatchNormV3/ReadVariableOpReadVariableOpGmodel_2_batch_normalization_10_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02@
>model_2/batch_normalization_10/FusedBatchNormV3/ReadVariableOpі
@model_2/batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpImodel_2_batch_normalization_10_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02B
@model_2/batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1а
/model_2/batch_normalization_10/FusedBatchNormV3FusedBatchNormV3"model_2/conv2d_11/BiasAdd:output:05model_2/batch_normalization_10/ReadVariableOp:value:07model_2/batch_normalization_10/ReadVariableOp_1:value:0Fmodel_2/batch_normalization_10/FusedBatchNormV3/ReadVariableOp:value:0Hmodel_2/batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:          : : : : :*
epsilon%oЃ:*
is_training( 21
/model_2/batch_normalization_10/FusedBatchNormV3»
model_2/activation_13/ReluRelu3model_2/batch_normalization_10/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:          2
model_2/activation_13/Reluй
model_2/add_4/addAddV2(model_2/activation_12/Relu:activations:0(model_2/activation_13/Relu:activations:0*
T0*/
_output_shapes
:          2
model_2/add_4/add╦
'model_2/conv2d_12/Conv2D/ReadVariableOpReadVariableOp0model_2_conv2d_12_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02)
'model_2/conv2d_12/Conv2D/ReadVariableOpУ
model_2/conv2d_12/Conv2DConv2Dmodel_2/add_4/add:z:0/model_2/conv2d_12/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:          *
paddingSAME*
strides
2
model_2/conv2d_12/Conv2D┬
(model_2/conv2d_12/BiasAdd/ReadVariableOpReadVariableOp1model_2_conv2d_12_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02*
(model_2/conv2d_12/BiasAdd/ReadVariableOpл
model_2/conv2d_12/BiasAddBiasAdd!model_2/conv2d_12/Conv2D:output:00model_2/conv2d_12/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          2
model_2/conv2d_12/BiasAddЛ
-model_2/batch_normalization_11/ReadVariableOpReadVariableOp6model_2_batch_normalization_11_readvariableop_resource*
_output_shapes
: *
dtype02/
-model_2/batch_normalization_11/ReadVariableOpО
/model_2/batch_normalization_11/ReadVariableOp_1ReadVariableOp8model_2_batch_normalization_11_readvariableop_1_resource*
_output_shapes
: *
dtype021
/model_2/batch_normalization_11/ReadVariableOp_1ё
>model_2/batch_normalization_11/FusedBatchNormV3/ReadVariableOpReadVariableOpGmodel_2_batch_normalization_11_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02@
>model_2/batch_normalization_11/FusedBatchNormV3/ReadVariableOpі
@model_2/batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpImodel_2_batch_normalization_11_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02B
@model_2/batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1а
/model_2/batch_normalization_11/FusedBatchNormV3FusedBatchNormV3"model_2/conv2d_12/BiasAdd:output:05model_2/batch_normalization_11/ReadVariableOp:value:07model_2/batch_normalization_11/ReadVariableOp_1:value:0Fmodel_2/batch_normalization_11/FusedBatchNormV3/ReadVariableOp:value:0Hmodel_2/batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:          : : : : :*
epsilon%oЃ:*
is_training( 21
/model_2/batch_normalization_11/FusedBatchNormV3»
model_2/activation_14/ReluRelu3model_2/batch_normalization_11/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:          2
model_2/activation_14/Reluф
model_2/add_5/addAddV2model_2/add_4/add:z:0(model_2/activation_14/Relu:activations:0*
T0*/
_output_shapes
:          2
model_2/add_5/addЉ
model_2/activation_15/ReluRelumodel_2/add_5/add:z:0*
T0*/
_output_shapes
:          2
model_2/activation_15/Reluќ
model_2/up_sampling2d_2/ShapeShape(model_2/activation_15/Relu:activations:0*
T0*
_output_shapes
:2
model_2/up_sampling2d_2/Shapeц
+model_2/up_sampling2d_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2-
+model_2/up_sampling2d_2/strided_slice/stackе
-model_2/up_sampling2d_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-model_2/up_sampling2d_2/strided_slice/stack_1е
-model_2/up_sampling2d_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-model_2/up_sampling2d_2/strided_slice/stack_2я
%model_2/up_sampling2d_2/strided_sliceStridedSlice&model_2/up_sampling2d_2/Shape:output:04model_2/up_sampling2d_2/strided_slice/stack:output:06model_2/up_sampling2d_2/strided_slice/stack_1:output:06model_2/up_sampling2d_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2'
%model_2/up_sampling2d_2/strided_sliceЈ
model_2/up_sampling2d_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
model_2/up_sampling2d_2/ConstЙ
model_2/up_sampling2d_2/mulMul.model_2/up_sampling2d_2/strided_slice:output:0&model_2/up_sampling2d_2/Const:output:0*
T0*
_output_shapes
:2
model_2/up_sampling2d_2/mulц
4model_2/up_sampling2d_2/resize/ResizeNearestNeighborResizeNearestNeighbor(model_2/activation_15/Relu:activations:0model_2/up_sampling2d_2/mul:z:0*
T0*/
_output_shapes
:          *
half_pixel_centers(26
4model_2/up_sampling2d_2/resize/ResizeNearestNeighbor╦
'model_2/conv2d_13/Conv2D/ReadVariableOpReadVariableOp0model_2_conv2d_13_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02)
'model_2/conv2d_13/Conv2D/ReadVariableOpў
model_2/conv2d_13/Conv2DConv2DEmodel_2/up_sampling2d_2/resize/ResizeNearestNeighbor:resized_images:0/model_2/conv2d_13/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:          *
paddingSAME*
strides
2
model_2/conv2d_13/Conv2D┬
(model_2/conv2d_13/BiasAdd/ReadVariableOpReadVariableOp1model_2_conv2d_13_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02*
(model_2/conv2d_13/BiasAdd/ReadVariableOpл
model_2/conv2d_13/BiasAddBiasAdd!model_2/conv2d_13/Conv2D:output:00model_2/conv2d_13/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          2
model_2/conv2d_13/BiasAddЛ
-model_2/batch_normalization_12/ReadVariableOpReadVariableOp6model_2_batch_normalization_12_readvariableop_resource*
_output_shapes
: *
dtype02/
-model_2/batch_normalization_12/ReadVariableOpО
/model_2/batch_normalization_12/ReadVariableOp_1ReadVariableOp8model_2_batch_normalization_12_readvariableop_1_resource*
_output_shapes
: *
dtype021
/model_2/batch_normalization_12/ReadVariableOp_1ё
>model_2/batch_normalization_12/FusedBatchNormV3/ReadVariableOpReadVariableOpGmodel_2_batch_normalization_12_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02@
>model_2/batch_normalization_12/FusedBatchNormV3/ReadVariableOpі
@model_2/batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpImodel_2_batch_normalization_12_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02B
@model_2/batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1а
/model_2/batch_normalization_12/FusedBatchNormV3FusedBatchNormV3"model_2/conv2d_13/BiasAdd:output:05model_2/batch_normalization_12/ReadVariableOp:value:07model_2/batch_normalization_12/ReadVariableOp_1:value:0Fmodel_2/batch_normalization_12/FusedBatchNormV3/ReadVariableOp:value:0Hmodel_2/batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:          : : : : :*
epsilon%oЃ:*
is_training( 21
/model_2/batch_normalization_12/FusedBatchNormV3»
model_2/activation_16/ReluRelu3model_2/batch_normalization_12/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:          2
model_2/activation_16/Relu╦
'model_2/conv2d_14/Conv2D/ReadVariableOpReadVariableOp0model_2_conv2d_14_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02)
'model_2/conv2d_14/Conv2D/ReadVariableOpч
model_2/conv2d_14/Conv2DConv2D(model_2/activation_16/Relu:activations:0/model_2/conv2d_14/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:          *
paddingSAME*
strides
2
model_2/conv2d_14/Conv2D┬
(model_2/conv2d_14/BiasAdd/ReadVariableOpReadVariableOp1model_2_conv2d_14_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02*
(model_2/conv2d_14/BiasAdd/ReadVariableOpл
model_2/conv2d_14/BiasAddBiasAdd!model_2/conv2d_14/Conv2D:output:00model_2/conv2d_14/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          2
model_2/conv2d_14/BiasAddЛ
-model_2/batch_normalization_13/ReadVariableOpReadVariableOp6model_2_batch_normalization_13_readvariableop_resource*
_output_shapes
: *
dtype02/
-model_2/batch_normalization_13/ReadVariableOpО
/model_2/batch_normalization_13/ReadVariableOp_1ReadVariableOp8model_2_batch_normalization_13_readvariableop_1_resource*
_output_shapes
: *
dtype021
/model_2/batch_normalization_13/ReadVariableOp_1ё
>model_2/batch_normalization_13/FusedBatchNormV3/ReadVariableOpReadVariableOpGmodel_2_batch_normalization_13_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02@
>model_2/batch_normalization_13/FusedBatchNormV3/ReadVariableOpі
@model_2/batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpImodel_2_batch_normalization_13_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02B
@model_2/batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1а
/model_2/batch_normalization_13/FusedBatchNormV3FusedBatchNormV3"model_2/conv2d_14/BiasAdd:output:05model_2/batch_normalization_13/ReadVariableOp:value:07model_2/batch_normalization_13/ReadVariableOp_1:value:0Fmodel_2/batch_normalization_13/FusedBatchNormV3/ReadVariableOp:value:0Hmodel_2/batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:          : : : : :*
epsilon%oЃ:*
is_training( 21
/model_2/batch_normalization_13/FusedBatchNormV3»
model_2/activation_17/ReluRelu3model_2/batch_normalization_13/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:          2
model_2/activation_17/Reluй
model_2/add_6/addAddV2(model_2/activation_16/Relu:activations:0(model_2/activation_17/Relu:activations:0*
T0*/
_output_shapes
:          2
model_2/add_6/add╦
'model_2/conv2d_15/Conv2D/ReadVariableOpReadVariableOp0model_2_conv2d_15_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02)
'model_2/conv2d_15/Conv2D/ReadVariableOpУ
model_2/conv2d_15/Conv2DConv2Dmodel_2/add_6/add:z:0/model_2/conv2d_15/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:          *
paddingSAME*
strides
2
model_2/conv2d_15/Conv2D┬
(model_2/conv2d_15/BiasAdd/ReadVariableOpReadVariableOp1model_2_conv2d_15_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02*
(model_2/conv2d_15/BiasAdd/ReadVariableOpл
model_2/conv2d_15/BiasAddBiasAdd!model_2/conv2d_15/Conv2D:output:00model_2/conv2d_15/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          2
model_2/conv2d_15/BiasAddЛ
-model_2/batch_normalization_14/ReadVariableOpReadVariableOp6model_2_batch_normalization_14_readvariableop_resource*
_output_shapes
: *
dtype02/
-model_2/batch_normalization_14/ReadVariableOpО
/model_2/batch_normalization_14/ReadVariableOp_1ReadVariableOp8model_2_batch_normalization_14_readvariableop_1_resource*
_output_shapes
: *
dtype021
/model_2/batch_normalization_14/ReadVariableOp_1ё
>model_2/batch_normalization_14/FusedBatchNormV3/ReadVariableOpReadVariableOpGmodel_2_batch_normalization_14_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02@
>model_2/batch_normalization_14/FusedBatchNormV3/ReadVariableOpі
@model_2/batch_normalization_14/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpImodel_2_batch_normalization_14_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02B
@model_2/batch_normalization_14/FusedBatchNormV3/ReadVariableOp_1а
/model_2/batch_normalization_14/FusedBatchNormV3FusedBatchNormV3"model_2/conv2d_15/BiasAdd:output:05model_2/batch_normalization_14/ReadVariableOp:value:07model_2/batch_normalization_14/ReadVariableOp_1:value:0Fmodel_2/batch_normalization_14/FusedBatchNormV3/ReadVariableOp:value:0Hmodel_2/batch_normalization_14/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:          : : : : :*
epsilon%oЃ:*
is_training( 21
/model_2/batch_normalization_14/FusedBatchNormV3»
model_2/activation_18/ReluRelu3model_2/batch_normalization_14/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:          2
model_2/activation_18/Reluф
model_2/add_7/addAddV2model_2/add_6/add:z:0(model_2/activation_18/Relu:activations:0*
T0*/
_output_shapes
:          2
model_2/add_7/addЉ
model_2/activation_19/ReluRelumodel_2/add_7/add:z:0*
T0*/
_output_shapes
:          2
model_2/activation_19/Reluќ
model_2/up_sampling2d_3/ShapeShape(model_2/activation_19/Relu:activations:0*
T0*
_output_shapes
:2
model_2/up_sampling2d_3/Shapeц
+model_2/up_sampling2d_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2-
+model_2/up_sampling2d_3/strided_slice/stackе
-model_2/up_sampling2d_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-model_2/up_sampling2d_3/strided_slice/stack_1е
-model_2/up_sampling2d_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-model_2/up_sampling2d_3/strided_slice/stack_2я
%model_2/up_sampling2d_3/strided_sliceStridedSlice&model_2/up_sampling2d_3/Shape:output:04model_2/up_sampling2d_3/strided_slice/stack:output:06model_2/up_sampling2d_3/strided_slice/stack_1:output:06model_2/up_sampling2d_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2'
%model_2/up_sampling2d_3/strided_sliceЈ
model_2/up_sampling2d_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
model_2/up_sampling2d_3/ConstЙ
model_2/up_sampling2d_3/mulMul.model_2/up_sampling2d_3/strided_slice:output:0&model_2/up_sampling2d_3/Const:output:0*
T0*
_output_shapes
:2
model_2/up_sampling2d_3/mulц
4model_2/up_sampling2d_3/resize/ResizeNearestNeighborResizeNearestNeighbor(model_2/activation_19/Relu:activations:0model_2/up_sampling2d_3/mul:z:0*
T0*/
_output_shapes
:          *
half_pixel_centers(26
4model_2/up_sampling2d_3/resize/ResizeNearestNeighbor╦
'model_2/conv2d_16/Conv2D/ReadVariableOpReadVariableOp0model_2_conv2d_16_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02)
'model_2/conv2d_16/Conv2D/ReadVariableOpў
model_2/conv2d_16/Conv2DConv2DEmodel_2/up_sampling2d_3/resize/ResizeNearestNeighbor:resized_images:0/model_2/conv2d_16/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:          *
paddingSAME*
strides
2
model_2/conv2d_16/Conv2D┬
(model_2/conv2d_16/BiasAdd/ReadVariableOpReadVariableOp1model_2_conv2d_16_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02*
(model_2/conv2d_16/BiasAdd/ReadVariableOpл
model_2/conv2d_16/BiasAddBiasAdd!model_2/conv2d_16/Conv2D:output:00model_2/conv2d_16/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          2
model_2/conv2d_16/BiasAddЛ
-model_2/batch_normalization_15/ReadVariableOpReadVariableOp6model_2_batch_normalization_15_readvariableop_resource*
_output_shapes
: *
dtype02/
-model_2/batch_normalization_15/ReadVariableOpО
/model_2/batch_normalization_15/ReadVariableOp_1ReadVariableOp8model_2_batch_normalization_15_readvariableop_1_resource*
_output_shapes
: *
dtype021
/model_2/batch_normalization_15/ReadVariableOp_1ё
>model_2/batch_normalization_15/FusedBatchNormV3/ReadVariableOpReadVariableOpGmodel_2_batch_normalization_15_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02@
>model_2/batch_normalization_15/FusedBatchNormV3/ReadVariableOpі
@model_2/batch_normalization_15/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpImodel_2_batch_normalization_15_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02B
@model_2/batch_normalization_15/FusedBatchNormV3/ReadVariableOp_1а
/model_2/batch_normalization_15/FusedBatchNormV3FusedBatchNormV3"model_2/conv2d_16/BiasAdd:output:05model_2/batch_normalization_15/ReadVariableOp:value:07model_2/batch_normalization_15/ReadVariableOp_1:value:0Fmodel_2/batch_normalization_15/FusedBatchNormV3/ReadVariableOp:value:0Hmodel_2/batch_normalization_15/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:          : : : : :*
epsilon%oЃ:*
is_training( 21
/model_2/batch_normalization_15/FusedBatchNormV3»
model_2/activation_20/ReluRelu3model_2/batch_normalization_15/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:          2
model_2/activation_20/Relu╦
'model_2/conv2d_17/Conv2D/ReadVariableOpReadVariableOp0model_2_conv2d_17_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02)
'model_2/conv2d_17/Conv2D/ReadVariableOpч
model_2/conv2d_17/Conv2DConv2D(model_2/activation_20/Relu:activations:0/model_2/conv2d_17/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
paddingSAME*
strides
2
model_2/conv2d_17/Conv2D┬
(model_2/conv2d_17/BiasAdd/ReadVariableOpReadVariableOp1model_2_conv2d_17_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(model_2/conv2d_17/BiasAdd/ReadVariableOpл
model_2/conv2d_17/BiasAddBiasAdd!model_2/conv2d_17/Conv2D:output:00model_2/conv2d_17/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         2
model_2/conv2d_17/BiasAddЛ
-model_2/batch_normalization_16/ReadVariableOpReadVariableOp6model_2_batch_normalization_16_readvariableop_resource*
_output_shapes
:*
dtype02/
-model_2/batch_normalization_16/ReadVariableOpО
/model_2/batch_normalization_16/ReadVariableOp_1ReadVariableOp8model_2_batch_normalization_16_readvariableop_1_resource*
_output_shapes
:*
dtype021
/model_2/batch_normalization_16/ReadVariableOp_1ё
>model_2/batch_normalization_16/FusedBatchNormV3/ReadVariableOpReadVariableOpGmodel_2_batch_normalization_16_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02@
>model_2/batch_normalization_16/FusedBatchNormV3/ReadVariableOpі
@model_2/batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpImodel_2_batch_normalization_16_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02B
@model_2/batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1а
/model_2/batch_normalization_16/FusedBatchNormV3FusedBatchNormV3"model_2/conv2d_17/BiasAdd:output:05model_2/batch_normalization_16/ReadVariableOp:value:07model_2/batch_normalization_16/ReadVariableOp_1:value:0Fmodel_2/batch_normalization_16/FusedBatchNormV3/ReadVariableOp:value:0Hmodel_2/batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         :::::*
epsilon%oЃ:*
is_training( 21
/model_2/batch_normalization_16/FusedBatchNormV3»
model_2/activation_21/ReluRelu3model_2/batch_normalization_16/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:         2
model_2/activation_21/Reluё
IdentityIdentity(model_2/activation_21/Relu:activations:0*
T0*/
_output_shapes
:         2

Identity"
identityIdentity:output:0*­
_input_shapesя
█:         :::::::::::::::::::::::::::::::::::::::::::::::::X T
/
_output_shapes
:         
!
_user_specified_name	input_3:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
: :!

_output_shapes
: :"

_output_shapes
: :#

_output_shapes
: :$

_output_shapes
: :%

_output_shapes
: :&

_output_shapes
: :'

_output_shapes
: :(

_output_shapes
: :)

_output_shapes
: :*

_output_shapes
: :+

_output_shapes
: :,

_output_shapes
: :-

_output_shapes
: :.

_output_shapes
: :/

_output_shapes
: :0

_output_shapes
: "»L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*└
serving_defaultг
C
input_38
serving_default_input_3:0         I
activation_218
StatefulPartitionedCall:0         tensorflow/serving/predict:╩┴
Иы
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer-6
layer-7
	layer_with_weights-4
	layer-8

layer_with_weights-5

layer-9
layer-10
layer-11
layer-12
layer-13
layer_with_weights-6
layer-14
layer_with_weights-7
layer-15
layer-16
layer_with_weights-8
layer-17
layer_with_weights-9
layer-18
layer-19
layer-20
layer_with_weights-10
layer-21
layer_with_weights-11
layer-22
layer-23
layer-24
layer-25
layer-26
layer_with_weights-12
layer-27
layer_with_weights-13
layer-28
layer-29
layer_with_weights-14
layer-30
 layer_with_weights-15
 layer-31
!layer-32
"regularization_losses
#trainable_variables
$	variables
%	keras_api
&
signatures
ё_default_save_signature
+Ё&call_and_return_all_conditional_losses
є__call__"јж
_tf_keras_modelзУ{"class_name": "Model", "name": "model_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "model_2", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 7, 7, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_3"}, "name": "input_3", "inbound_nodes": []}, {"class_name": "Conv2D", "config": {"name": "conv2d_10", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_10", "inbound_nodes": [[["input_3", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_9", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.5, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_9", "inbound_nodes": [[["conv2d_10", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_12", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_12", "inbound_nodes": [[["batch_normalization_9", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_11", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_11", "inbound_nodes": [[["activation_12", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_10", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.5, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_10", "inbound_nodes": [[["conv2d_11", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_13", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_13", "inbound_nodes": [[["batch_normalization_10", 0, 0, {}]]]}, {"class_name": "Add", "config": {"name": "add_4", "trainable": true, "dtype": "float32"}, "name": "add_4", "inbound_nodes": [[["activation_12", 0, 0, {}], ["activation_13", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_12", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_12", "inbound_nodes": [[["add_4", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_11", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.5, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_11", "inbound_nodes": [[["conv2d_12", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_14", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_14", "inbound_nodes": [[["batch_normalization_11", 0, 0, {}]]]}, {"class_name": "Add", "config": {"name": "add_5", "trainable": true, "dtype": "float32"}, "name": "add_5", "inbound_nodes": [[["add_4", 0, 0, {}], ["activation_14", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_15", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_15", "inbound_nodes": [[["add_5", 0, 0, {}]]]}, {"class_name": "UpSampling2D", "config": {"name": "up_sampling2d_2", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "name": "up_sampling2d_2", "inbound_nodes": [[["activation_15", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_13", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_13", "inbound_nodes": [[["up_sampling2d_2", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_12", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.5, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_12", "inbound_nodes": [[["conv2d_13", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_16", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_16", "inbound_nodes": [[["batch_normalization_12", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_14", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_14", "inbound_nodes": [[["activation_16", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_13", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.5, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_13", "inbound_nodes": [[["conv2d_14", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_17", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_17", "inbound_nodes": [[["batch_normalization_13", 0, 0, {}]]]}, {"class_name": "Add", "config": {"name": "add_6", "trainable": true, "dtype": "float32"}, "name": "add_6", "inbound_nodes": [[["activation_16", 0, 0, {}], ["activation_17", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_15", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_15", "inbound_nodes": [[["add_6", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_14", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.5, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_14", "inbound_nodes": [[["conv2d_15", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_18", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_18", "inbound_nodes": [[["batch_normalization_14", 0, 0, {}]]]}, {"class_name": "Add", "config": {"name": "add_7", "trainable": true, "dtype": "float32"}, "name": "add_7", "inbound_nodes": [[["add_6", 0, 0, {}], ["activation_18", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_19", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_19", "inbound_nodes": [[["add_7", 0, 0, {}]]]}, {"class_name": "UpSampling2D", "config": {"name": "up_sampling2d_3", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "name": "up_sampling2d_3", "inbound_nodes": [[["activation_19", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_16", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_16", "inbound_nodes": [[["up_sampling2d_3", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_15", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.5, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_15", "inbound_nodes": [[["conv2d_16", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_20", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_20", "inbound_nodes": [[["batch_normalization_15", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_17", "trainable": true, "dtype": "float32", "filters": 1, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_17", "inbound_nodes": [[["activation_20", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_16", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.5, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_16", "inbound_nodes": [[["conv2d_17", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_21", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_21", "inbound_nodes": [[["batch_normalization_16", 0, 0, {}]]]}], "input_layers": [["input_3", 0, 0]], "output_layers": [["activation_21", 0, 0]]}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 7, 7, 1]}, "is_graph_network": true, "keras_version": "2.3.0-tf", "backend": "tensorflow", "model_config": {"class_name": "Model", "config": {"name": "model_2", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 7, 7, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_3"}, "name": "input_3", "inbound_nodes": []}, {"class_name": "Conv2D", "config": {"name": "conv2d_10", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_10", "inbound_nodes": [[["input_3", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_9", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.5, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_9", "inbound_nodes": [[["conv2d_10", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_12", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_12", "inbound_nodes": [[["batch_normalization_9", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_11", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_11", "inbound_nodes": [[["activation_12", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_10", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.5, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_10", "inbound_nodes": [[["conv2d_11", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_13", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_13", "inbound_nodes": [[["batch_normalization_10", 0, 0, {}]]]}, {"class_name": "Add", "config": {"name": "add_4", "trainable": true, "dtype": "float32"}, "name": "add_4", "inbound_nodes": [[["activation_12", 0, 0, {}], ["activation_13", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_12", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_12", "inbound_nodes": [[["add_4", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_11", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.5, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_11", "inbound_nodes": [[["conv2d_12", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_14", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_14", "inbound_nodes": [[["batch_normalization_11", 0, 0, {}]]]}, {"class_name": "Add", "config": {"name": "add_5", "trainable": true, "dtype": "float32"}, "name": "add_5", "inbound_nodes": [[["add_4", 0, 0, {}], ["activation_14", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_15", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_15", "inbound_nodes": [[["add_5", 0, 0, {}]]]}, {"class_name": "UpSampling2D", "config": {"name": "up_sampling2d_2", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "name": "up_sampling2d_2", "inbound_nodes": [[["activation_15", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_13", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_13", "inbound_nodes": [[["up_sampling2d_2", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_12", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.5, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_12", "inbound_nodes": [[["conv2d_13", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_16", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_16", "inbound_nodes": [[["batch_normalization_12", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_14", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_14", "inbound_nodes": [[["activation_16", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_13", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.5, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_13", "inbound_nodes": [[["conv2d_14", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_17", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_17", "inbound_nodes": [[["batch_normalization_13", 0, 0, {}]]]}, {"class_name": "Add", "config": {"name": "add_6", "trainable": true, "dtype": "float32"}, "name": "add_6", "inbound_nodes": [[["activation_16", 0, 0, {}], ["activation_17", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_15", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_15", "inbound_nodes": [[["add_6", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_14", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.5, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_14", "inbound_nodes": [[["conv2d_15", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_18", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_18", "inbound_nodes": [[["batch_normalization_14", 0, 0, {}]]]}, {"class_name": "Add", "config": {"name": "add_7", "trainable": true, "dtype": "float32"}, "name": "add_7", "inbound_nodes": [[["add_6", 0, 0, {}], ["activation_18", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_19", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_19", "inbound_nodes": [[["add_7", 0, 0, {}]]]}, {"class_name": "UpSampling2D", "config": {"name": "up_sampling2d_3", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "name": "up_sampling2d_3", "inbound_nodes": [[["activation_19", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_16", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_16", "inbound_nodes": [[["up_sampling2d_3", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_15", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.5, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_15", "inbound_nodes": [[["conv2d_16", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_20", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_20", "inbound_nodes": [[["batch_normalization_15", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_17", "trainable": true, "dtype": "float32", "filters": 1, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_17", "inbound_nodes": [[["activation_20", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_16", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.5, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_16", "inbound_nodes": [[["conv2d_17", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_21", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_21", "inbound_nodes": [[["batch_normalization_16", 0, 0, {}]]]}], "input_layers": [["input_3", 0, 0]], "output_layers": [["activation_21", 0, 0]]}}}
ш"Ы
_tf_keras_input_layerм{"class_name": "InputLayer", "name": "input_3", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 7, 7, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 7, 7, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_3"}}
─	

'kernel
(bias
)regularization_losses
*trainable_variables
+	variables
,	keras_api
+Є&call_and_return_all_conditional_losses
ѕ__call__"Ю
_tf_keras_layerЃ{"class_name": "Conv2D", "name": "conv2d_10", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "conv2d_10", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 1}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 7, 7, 1]}}
ќ	
-axis
	.gamma
/beta
0moving_mean
1moving_variance
2regularization_losses
3trainable_variables
4	variables
5	keras_api
+Ѕ&call_and_return_all_conditional_losses
і__call__"└
_tf_keras_layerд{"class_name": "BatchNormalization", "name": "batch_normalization_9", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "batch_normalization_9", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.5, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 7, 7, 32]}}
Х
6regularization_losses
7trainable_variables
8	variables
9	keras_api
+І&call_and_return_all_conditional_losses
ї__call__"Ц
_tf_keras_layerІ{"class_name": "Activation", "name": "activation_12", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "activation_12", "trainable": true, "dtype": "float32", "activation": "relu"}}
к	

:kernel
;bias
<regularization_losses
=trainable_variables
>	variables
?	keras_api
+Ї&call_and_return_all_conditional_losses
ј__call__"Ъ
_tf_keras_layerЁ{"class_name": "Conv2D", "name": "conv2d_11", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "conv2d_11", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 7, 7, 32]}}
ў	
@axis
	Agamma
Bbeta
Cmoving_mean
Dmoving_variance
Eregularization_losses
Ftrainable_variables
G	variables
H	keras_api
+Ј&call_and_return_all_conditional_losses
љ__call__"┬
_tf_keras_layerе{"class_name": "BatchNormalization", "name": "batch_normalization_10", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "batch_normalization_10", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.5, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 7, 7, 32]}}
Х
Iregularization_losses
Jtrainable_variables
K	variables
L	keras_api
+Љ&call_and_return_all_conditional_losses
њ__call__"Ц
_tf_keras_layerІ{"class_name": "Activation", "name": "activation_13", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "activation_13", "trainable": true, "dtype": "float32", "activation": "relu"}}
ћ
Mregularization_losses
Ntrainable_variables
O	variables
P	keras_api
+Њ&call_and_return_all_conditional_losses
ћ__call__"Ѓ
_tf_keras_layerж{"class_name": "Add", "name": "add_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "add_4", "trainable": true, "dtype": "float32"}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 7, 7, 32]}, {"class_name": "TensorShape", "items": [null, 7, 7, 32]}]}
к	

Qkernel
Rbias
Sregularization_losses
Ttrainable_variables
U	variables
V	keras_api
+Ћ&call_and_return_all_conditional_losses
ќ__call__"Ъ
_tf_keras_layerЁ{"class_name": "Conv2D", "name": "conv2d_12", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "conv2d_12", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 7, 7, 32]}}
ў	
Waxis
	Xgamma
Ybeta
Zmoving_mean
[moving_variance
\regularization_losses
]trainable_variables
^	variables
_	keras_api
+Ќ&call_and_return_all_conditional_losses
ў__call__"┬
_tf_keras_layerе{"class_name": "BatchNormalization", "name": "batch_normalization_11", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "batch_normalization_11", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.5, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 7, 7, 32]}}
Х
`regularization_losses
atrainable_variables
b	variables
c	keras_api
+Ў&call_and_return_all_conditional_losses
џ__call__"Ц
_tf_keras_layerІ{"class_name": "Activation", "name": "activation_14", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "activation_14", "trainable": true, "dtype": "float32", "activation": "relu"}}
ћ
dregularization_losses
etrainable_variables
f	variables
g	keras_api
+Џ&call_and_return_all_conditional_losses
ю__call__"Ѓ
_tf_keras_layerж{"class_name": "Add", "name": "add_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "add_5", "trainable": true, "dtype": "float32"}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 7, 7, 32]}, {"class_name": "TensorShape", "items": [null, 7, 7, 32]}]}
Х
hregularization_losses
itrainable_variables
j	variables
k	keras_api
+Ю&call_and_return_all_conditional_losses
ъ__call__"Ц
_tf_keras_layerІ{"class_name": "Activation", "name": "activation_15", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "activation_15", "trainable": true, "dtype": "float32", "activation": "relu"}}
е
lregularization_losses
mtrainable_variables
n	variables
o	keras_api
+Ъ&call_and_return_all_conditional_losses
а__call__"Ќ
_tf_keras_layer§{"class_name": "UpSampling2D", "name": "up_sampling2d_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "up_sampling2d_2", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
╚	

pkernel
qbias
rregularization_losses
strainable_variables
t	variables
u	keras_api
+А&call_and_return_all_conditional_losses
б__call__"А
_tf_keras_layerЄ{"class_name": "Conv2D", "name": "conv2d_13", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "conv2d_13", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 14, 14, 32]}}
џ	
vaxis
	wgamma
xbeta
ymoving_mean
zmoving_variance
{regularization_losses
|trainable_variables
}	variables
~	keras_api
+Б&call_and_return_all_conditional_losses
ц__call__"─
_tf_keras_layerф{"class_name": "BatchNormalization", "name": "batch_normalization_12", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "batch_normalization_12", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.5, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 14, 14, 32]}}
╣
regularization_losses
ђtrainable_variables
Ђ	variables
ѓ	keras_api
+Ц&call_and_return_all_conditional_losses
д__call__"Ц
_tf_keras_layerІ{"class_name": "Activation", "name": "activation_16", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "activation_16", "trainable": true, "dtype": "float32", "activation": "relu"}}
╬	
Ѓkernel
	ёbias
Ёregularization_losses
єtrainable_variables
Є	variables
ѕ	keras_api
+Д&call_and_return_all_conditional_losses
е__call__"А
_tf_keras_layerЄ{"class_name": "Conv2D", "name": "conv2d_14", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "conv2d_14", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 14, 14, 32]}}
Б	
	Ѕaxis

іgamma
	Іbeta
їmoving_mean
Їmoving_variance
јregularization_losses
Јtrainable_variables
љ	variables
Љ	keras_api
+Е&call_and_return_all_conditional_losses
ф__call__"─
_tf_keras_layerф{"class_name": "BatchNormalization", "name": "batch_normalization_13", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "batch_normalization_13", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.5, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 14, 14, 32]}}
║
њregularization_losses
Њtrainable_variables
ћ	variables
Ћ	keras_api
+Ф&call_and_return_all_conditional_losses
г__call__"Ц
_tf_keras_layerІ{"class_name": "Activation", "name": "activation_17", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "activation_17", "trainable": true, "dtype": "float32", "activation": "relu"}}
ю
ќregularization_losses
Ќtrainable_variables
ў	variables
Ў	keras_api
+Г&call_and_return_all_conditional_losses
«__call__"Є
_tf_keras_layerь{"class_name": "Add", "name": "add_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "add_6", "trainable": true, "dtype": "float32"}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 14, 14, 32]}, {"class_name": "TensorShape", "items": [null, 14, 14, 32]}]}
╬	
џkernel
	Џbias
юregularization_losses
Юtrainable_variables
ъ	variables
Ъ	keras_api
+»&call_and_return_all_conditional_losses
░__call__"А
_tf_keras_layerЄ{"class_name": "Conv2D", "name": "conv2d_15", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "conv2d_15", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 14, 14, 32]}}
Б	
	аaxis

Аgamma
	бbeta
Бmoving_mean
цmoving_variance
Цregularization_losses
дtrainable_variables
Д	variables
е	keras_api
+▒&call_and_return_all_conditional_losses
▓__call__"─
_tf_keras_layerф{"class_name": "BatchNormalization", "name": "batch_normalization_14", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "batch_normalization_14", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.5, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 14, 14, 32]}}
║
Еregularization_losses
фtrainable_variables
Ф	variables
г	keras_api
+│&call_and_return_all_conditional_losses
┤__call__"Ц
_tf_keras_layerІ{"class_name": "Activation", "name": "activation_18", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "activation_18", "trainable": true, "dtype": "float32", "activation": "relu"}}
ю
Гregularization_losses
«trainable_variables
»	variables
░	keras_api
+х&call_and_return_all_conditional_losses
Х__call__"Є
_tf_keras_layerь{"class_name": "Add", "name": "add_7", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "add_7", "trainable": true, "dtype": "float32"}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 14, 14, 32]}, {"class_name": "TensorShape", "items": [null, 14, 14, 32]}]}
║
▒regularization_losses
▓trainable_variables
│	variables
┤	keras_api
+и&call_and_return_all_conditional_losses
И__call__"Ц
_tf_keras_layerІ{"class_name": "Activation", "name": "activation_19", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "activation_19", "trainable": true, "dtype": "float32", "activation": "relu"}}
г
хregularization_losses
Хtrainable_variables
и	variables
И	keras_api
+╣&call_and_return_all_conditional_losses
║__call__"Ќ
_tf_keras_layer§{"class_name": "UpSampling2D", "name": "up_sampling2d_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "up_sampling2d_3", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
╬	
╣kernel
	║bias
╗regularization_losses
╝trainable_variables
й	variables
Й	keras_api
+╗&call_and_return_all_conditional_losses
╝__call__"А
_tf_keras_layerЄ{"class_name": "Conv2D", "name": "conv2d_16", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "conv2d_16", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 28, 28, 32]}}
Б	
	┐axis

└gamma
	┴beta
┬moving_mean
├moving_variance
─regularization_losses
┼trainable_variables
к	variables
К	keras_api
+й&call_and_return_all_conditional_losses
Й__call__"─
_tf_keras_layerф{"class_name": "BatchNormalization", "name": "batch_normalization_15", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "batch_normalization_15", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.5, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 28, 28, 32]}}
║
╚regularization_losses
╔trainable_variables
╩	variables
╦	keras_api
+┐&call_and_return_all_conditional_losses
└__call__"Ц
_tf_keras_layerІ{"class_name": "Activation", "name": "activation_20", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "activation_20", "trainable": true, "dtype": "float32", "activation": "relu"}}
═	
╠kernel
	═bias
╬regularization_losses
¤trainable_variables
л	variables
Л	keras_api
+┴&call_and_return_all_conditional_losses
┬__call__"а
_tf_keras_layerє{"class_name": "Conv2D", "name": "conv2d_17", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "conv2d_17", "trainable": true, "dtype": "float32", "filters": 1, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 28, 28, 32]}}
А	
	мaxis

Мgamma
	нbeta
Нmoving_mean
оmoving_variance
Оregularization_losses
пtrainable_variables
┘	variables
┌	keras_api
+├&call_and_return_all_conditional_losses
─__call__"┬
_tf_keras_layerе{"class_name": "BatchNormalization", "name": "batch_normalization_16", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "batch_normalization_16", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.5, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 1}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 28, 28, 1]}}
║
█regularization_losses
▄trainable_variables
П	variables
я	keras_api
+┼&call_and_return_all_conditional_losses
к__call__"Ц
_tf_keras_layerІ{"class_name": "Activation", "name": "activation_21", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "activation_21", "trainable": true, "dtype": "float32", "activation": "relu"}}
 "
trackable_list_wrapper
д
'0
(1
.2
/3
:4
;5
A6
B7
Q8
R9
X10
Y11
p12
q13
w14
x15
Ѓ16
ё17
і18
І19
џ20
Џ21
А22
б23
╣24
║25
└26
┴27
╠28
═29
М30
н31"
trackable_list_wrapper
«
'0
(1
.2
/3
04
15
:6
;7
A8
B9
C10
D11
Q12
R13
X14
Y15
Z16
[17
p18
q19
w20
x21
y22
z23
Ѓ24
ё25
і26
І27
ї28
Ї29
џ30
Џ31
А32
б33
Б34
ц35
╣36
║37
└38
┴39
┬40
├41
╠42
═43
М44
н45
Н46
о47"
trackable_list_wrapper
М
 ▀layer_regularization_losses
"regularization_losses
Яmetrics
рlayers
Рnon_trainable_variables
#trainable_variables
сlayer_metrics
$	variables
є__call__
ё_default_save_signature
+Ё&call_and_return_all_conditional_losses
'Ё"call_and_return_conditional_losses"
_generic_user_object
-
Кserving_default"
signature_map
*:( 2conv2d_10/kernel
: 2conv2d_10/bias
 "
trackable_list_wrapper
.
'0
(1"
trackable_list_wrapper
.
'0
(1"
trackable_list_wrapper
х
 Сlayer_regularization_losses
тmetrics
Тlayers
)regularization_losses
уnon_trainable_variables
*trainable_variables
Уlayer_metrics
+	variables
ѕ__call__
+Є&call_and_return_all_conditional_losses
'Є"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
):' 2batch_normalization_9/gamma
(:& 2batch_normalization_9/beta
1:/  (2!batch_normalization_9/moving_mean
5:3  (2%batch_normalization_9/moving_variance
 "
trackable_list_wrapper
.
.0
/1"
trackable_list_wrapper
<
.0
/1
02
13"
trackable_list_wrapper
х
 жlayer_regularization_losses
Жmetrics
вlayers
2regularization_losses
Вnon_trainable_variables
3trainable_variables
ьlayer_metrics
4	variables
і__call__
+Ѕ&call_and_return_all_conditional_losses
'Ѕ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
х
 Ьlayer_regularization_losses
№metrics
­layers
6regularization_losses
ыnon_trainable_variables
7trainable_variables
Ыlayer_metrics
8	variables
ї__call__
+І&call_and_return_all_conditional_losses
'І"call_and_return_conditional_losses"
_generic_user_object
*:(  2conv2d_11/kernel
: 2conv2d_11/bias
 "
trackable_list_wrapper
.
:0
;1"
trackable_list_wrapper
.
:0
;1"
trackable_list_wrapper
х
 зlayer_regularization_losses
Зmetrics
шlayers
<regularization_losses
Шnon_trainable_variables
=trainable_variables
эlayer_metrics
>	variables
ј__call__
+Ї&call_and_return_all_conditional_losses
'Ї"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:( 2batch_normalization_10/gamma
):' 2batch_normalization_10/beta
2:0  (2"batch_normalization_10/moving_mean
6:4  (2&batch_normalization_10/moving_variance
 "
trackable_list_wrapper
.
A0
B1"
trackable_list_wrapper
<
A0
B1
C2
D3"
trackable_list_wrapper
х
 Эlayer_regularization_losses
щmetrics
Щlayers
Eregularization_losses
чnon_trainable_variables
Ftrainable_variables
Чlayer_metrics
G	variables
љ__call__
+Ј&call_and_return_all_conditional_losses
'Ј"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
х
 §layer_regularization_losses
■metrics
 layers
Iregularization_losses
ђnon_trainable_variables
Jtrainable_variables
Ђlayer_metrics
K	variables
њ__call__
+Љ&call_and_return_all_conditional_losses
'Љ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
х
 ѓlayer_regularization_losses
Ѓmetrics
ёlayers
Mregularization_losses
Ёnon_trainable_variables
Ntrainable_variables
єlayer_metrics
O	variables
ћ__call__
+Њ&call_and_return_all_conditional_losses
'Њ"call_and_return_conditional_losses"
_generic_user_object
*:(  2conv2d_12/kernel
: 2conv2d_12/bias
 "
trackable_list_wrapper
.
Q0
R1"
trackable_list_wrapper
.
Q0
R1"
trackable_list_wrapper
х
 Єlayer_regularization_losses
ѕmetrics
Ѕlayers
Sregularization_losses
іnon_trainable_variables
Ttrainable_variables
Іlayer_metrics
U	variables
ќ__call__
+Ћ&call_and_return_all_conditional_losses
'Ћ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:( 2batch_normalization_11/gamma
):' 2batch_normalization_11/beta
2:0  (2"batch_normalization_11/moving_mean
6:4  (2&batch_normalization_11/moving_variance
 "
trackable_list_wrapper
.
X0
Y1"
trackable_list_wrapper
<
X0
Y1
Z2
[3"
trackable_list_wrapper
х
 їlayer_regularization_losses
Їmetrics
јlayers
\regularization_losses
Јnon_trainable_variables
]trainable_variables
љlayer_metrics
^	variables
ў__call__
+Ќ&call_and_return_all_conditional_losses
'Ќ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
х
 Љlayer_regularization_losses
њmetrics
Њlayers
`regularization_losses
ћnon_trainable_variables
atrainable_variables
Ћlayer_metrics
b	variables
џ__call__
+Ў&call_and_return_all_conditional_losses
'Ў"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
х
 ќlayer_regularization_losses
Ќmetrics
ўlayers
dregularization_losses
Ўnon_trainable_variables
etrainable_variables
џlayer_metrics
f	variables
ю__call__
+Џ&call_and_return_all_conditional_losses
'Џ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
х
 Џlayer_regularization_losses
юmetrics
Юlayers
hregularization_losses
ъnon_trainable_variables
itrainable_variables
Ъlayer_metrics
j	variables
ъ__call__
+Ю&call_and_return_all_conditional_losses
'Ю"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
х
 аlayer_regularization_losses
Аmetrics
бlayers
lregularization_losses
Бnon_trainable_variables
mtrainable_variables
цlayer_metrics
n	variables
а__call__
+Ъ&call_and_return_all_conditional_losses
'Ъ"call_and_return_conditional_losses"
_generic_user_object
*:(  2conv2d_13/kernel
: 2conv2d_13/bias
 "
trackable_list_wrapper
.
p0
q1"
trackable_list_wrapper
.
p0
q1"
trackable_list_wrapper
х
 Цlayer_regularization_losses
дmetrics
Дlayers
rregularization_losses
еnon_trainable_variables
strainable_variables
Еlayer_metrics
t	variables
б__call__
+А&call_and_return_all_conditional_losses
'А"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:( 2batch_normalization_12/gamma
):' 2batch_normalization_12/beta
2:0  (2"batch_normalization_12/moving_mean
6:4  (2&batch_normalization_12/moving_variance
 "
trackable_list_wrapper
.
w0
x1"
trackable_list_wrapper
<
w0
x1
y2
z3"
trackable_list_wrapper
х
 фlayer_regularization_losses
Фmetrics
гlayers
{regularization_losses
Гnon_trainable_variables
|trainable_variables
«layer_metrics
}	variables
ц__call__
+Б&call_and_return_all_conditional_losses
'Б"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
и
 »layer_regularization_losses
░metrics
▒layers
regularization_losses
▓non_trainable_variables
ђtrainable_variables
│layer_metrics
Ђ	variables
д__call__
+Ц&call_and_return_all_conditional_losses
'Ц"call_and_return_conditional_losses"
_generic_user_object
*:(  2conv2d_14/kernel
: 2conv2d_14/bias
 "
trackable_list_wrapper
0
Ѓ0
ё1"
trackable_list_wrapper
0
Ѓ0
ё1"
trackable_list_wrapper
И
 ┤layer_regularization_losses
хmetrics
Хlayers
Ёregularization_losses
иnon_trainable_variables
єtrainable_variables
Иlayer_metrics
Є	variables
е__call__
+Д&call_and_return_all_conditional_losses
'Д"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:( 2batch_normalization_13/gamma
):' 2batch_normalization_13/beta
2:0  (2"batch_normalization_13/moving_mean
6:4  (2&batch_normalization_13/moving_variance
 "
trackable_list_wrapper
0
і0
І1"
trackable_list_wrapper
@
і0
І1
ї2
Ї3"
trackable_list_wrapper
И
 ╣layer_regularization_losses
║metrics
╗layers
јregularization_losses
╝non_trainable_variables
Јtrainable_variables
йlayer_metrics
љ	variables
ф__call__
+Е&call_and_return_all_conditional_losses
'Е"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
 Йlayer_regularization_losses
┐metrics
└layers
њregularization_losses
┴non_trainable_variables
Њtrainable_variables
┬layer_metrics
ћ	variables
г__call__
+Ф&call_and_return_all_conditional_losses
'Ф"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
 ├layer_regularization_losses
─metrics
┼layers
ќregularization_losses
кnon_trainable_variables
Ќtrainable_variables
Кlayer_metrics
ў	variables
«__call__
+Г&call_and_return_all_conditional_losses
'Г"call_and_return_conditional_losses"
_generic_user_object
*:(  2conv2d_15/kernel
: 2conv2d_15/bias
 "
trackable_list_wrapper
0
џ0
Џ1"
trackable_list_wrapper
0
џ0
Џ1"
trackable_list_wrapper
И
 ╚layer_regularization_losses
╔metrics
╩layers
юregularization_losses
╦non_trainable_variables
Юtrainable_variables
╠layer_metrics
ъ	variables
░__call__
+»&call_and_return_all_conditional_losses
'»"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:( 2batch_normalization_14/gamma
):' 2batch_normalization_14/beta
2:0  (2"batch_normalization_14/moving_mean
6:4  (2&batch_normalization_14/moving_variance
 "
trackable_list_wrapper
0
А0
б1"
trackable_list_wrapper
@
А0
б1
Б2
ц3"
trackable_list_wrapper
И
 ═layer_regularization_losses
╬metrics
¤layers
Цregularization_losses
лnon_trainable_variables
дtrainable_variables
Лlayer_metrics
Д	variables
▓__call__
+▒&call_and_return_all_conditional_losses
'▒"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
 мlayer_regularization_losses
Мmetrics
нlayers
Еregularization_losses
Нnon_trainable_variables
фtrainable_variables
оlayer_metrics
Ф	variables
┤__call__
+│&call_and_return_all_conditional_losses
'│"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
 Оlayer_regularization_losses
пmetrics
┘layers
Гregularization_losses
┌non_trainable_variables
«trainable_variables
█layer_metrics
»	variables
Х__call__
+х&call_and_return_all_conditional_losses
'х"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
 ▄layer_regularization_losses
Пmetrics
яlayers
▒regularization_losses
▀non_trainable_variables
▓trainable_variables
Яlayer_metrics
│	variables
И__call__
+и&call_and_return_all_conditional_losses
'и"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
 рlayer_regularization_losses
Рmetrics
сlayers
хregularization_losses
Сnon_trainable_variables
Хtrainable_variables
тlayer_metrics
и	variables
║__call__
+╣&call_and_return_all_conditional_losses
'╣"call_and_return_conditional_losses"
_generic_user_object
*:(  2conv2d_16/kernel
: 2conv2d_16/bias
 "
trackable_list_wrapper
0
╣0
║1"
trackable_list_wrapper
0
╣0
║1"
trackable_list_wrapper
И
 Тlayer_regularization_losses
уmetrics
Уlayers
╗regularization_losses
жnon_trainable_variables
╝trainable_variables
Жlayer_metrics
й	variables
╝__call__
+╗&call_and_return_all_conditional_losses
'╗"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:( 2batch_normalization_15/gamma
):' 2batch_normalization_15/beta
2:0  (2"batch_normalization_15/moving_mean
6:4  (2&batch_normalization_15/moving_variance
 "
trackable_list_wrapper
0
└0
┴1"
trackable_list_wrapper
@
└0
┴1
┬2
├3"
trackable_list_wrapper
И
 вlayer_regularization_losses
Вmetrics
ьlayers
─regularization_losses
Ьnon_trainable_variables
┼trainable_variables
№layer_metrics
к	variables
Й__call__
+й&call_and_return_all_conditional_losses
'й"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
 ­layer_regularization_losses
ыmetrics
Ыlayers
╚regularization_losses
зnon_trainable_variables
╔trainable_variables
Зlayer_metrics
╩	variables
└__call__
+┐&call_and_return_all_conditional_losses
'┐"call_and_return_conditional_losses"
_generic_user_object
*:( 2conv2d_17/kernel
:2conv2d_17/bias
 "
trackable_list_wrapper
0
╠0
═1"
trackable_list_wrapper
0
╠0
═1"
trackable_list_wrapper
И
 шlayer_regularization_losses
Шmetrics
эlayers
╬regularization_losses
Эnon_trainable_variables
¤trainable_variables
щlayer_metrics
л	variables
┬__call__
+┴&call_and_return_all_conditional_losses
'┴"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:(2batch_normalization_16/gamma
):'2batch_normalization_16/beta
2:0 (2"batch_normalization_16/moving_mean
6:4 (2&batch_normalization_16/moving_variance
 "
trackable_list_wrapper
0
М0
н1"
trackable_list_wrapper
@
М0
н1
Н2
о3"
trackable_list_wrapper
И
 Щlayer_regularization_losses
чmetrics
Чlayers
Оregularization_losses
§non_trainable_variables
пtrainable_variables
■layer_metrics
┘	variables
─__call__
+├&call_and_return_all_conditional_losses
'├"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
  layer_regularization_losses
ђmetrics
Ђlayers
█regularization_losses
ѓnon_trainable_variables
▄trainable_variables
Ѓlayer_metrics
П	variables
к__call__
+┼&call_and_return_all_conditional_losses
'┼"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
ъ
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
 31
!32"
trackable_list_wrapper
ъ
00
11
C2
D3
Z4
[5
y6
z7
ї8
Ї9
Б10
ц11
┬12
├13
Н14
о15"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
00
11"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
C0
D1"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
Z0
[1"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
y0
z1"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
ї0
Ї1"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
Б0
ц1"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
┬0
├1"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
Н0
о1"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
у2С
!__inference__wrapped_model_253196Й
І▓Є
FullArgSpec
argsџ 
varargsjargs
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *.б+
)і&
input_3         
┌2О
C__inference_model_2_layer_call_and_return_conditional_losses_256364
C__inference_model_2_layer_call_and_return_conditional_losses_255074
C__inference_model_2_layer_call_and_return_conditional_losses_255206
C__inference_model_2_layer_call_and_return_conditional_losses_256170└
и▓│
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
Ь2в
(__inference_model_2_layer_call_fn_256566
(__inference_model_2_layer_call_fn_256465
(__inference_model_2_layer_call_fn_255673
(__inference_model_2_layer_call_fn_255440└
и▓│
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
ц2А
E__inference_conv2d_10_layer_call_and_return_conditional_losses_253207О
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *7б4
2і/+                           
Ѕ2є
*__inference_conv2d_10_layer_call_fn_253217О
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *7б4
2і/+                           
є2Ѓ
Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_256609
Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_256684
Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_256702
Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_256627┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
џ2Ќ
6__inference_batch_normalization_9_layer_call_fn_256715
6__inference_batch_normalization_9_layer_call_fn_256728
6__inference_batch_normalization_9_layer_call_fn_256640
6__inference_batch_normalization_9_layer_call_fn_256653┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
з2­
I__inference_activation_12_layer_call_and_return_conditional_losses_256733б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
п2Н
.__inference_activation_12_layer_call_fn_256738б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ц2А
E__inference_conv2d_11_layer_call_and_return_conditional_losses_253354О
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *7б4
2і/+                            
Ѕ2є
*__inference_conv2d_11_layer_call_fn_253364О
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *7б4
2і/+                            
і2Є
R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_256874
R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_256856
R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_256781
R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_256799┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
ъ2Џ
7__inference_batch_normalization_10_layer_call_fn_256900
7__inference_batch_normalization_10_layer_call_fn_256825
7__inference_batch_normalization_10_layer_call_fn_256812
7__inference_batch_normalization_10_layer_call_fn_256887┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
з2­
I__inference_activation_13_layer_call_and_return_conditional_losses_256905б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
п2Н
.__inference_activation_13_layer_call_fn_256910б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
в2У
A__inference_add_4_layer_call_and_return_conditional_losses_256916б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
л2═
&__inference_add_4_layer_call_fn_256922б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ц2А
E__inference_conv2d_12_layer_call_and_return_conditional_losses_253501О
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *7б4
2і/+                            
Ѕ2є
*__inference_conv2d_12_layer_call_fn_253511О
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *7б4
2і/+                            
і2Є
R__inference_batch_normalization_11_layer_call_and_return_conditional_losses_257040
R__inference_batch_normalization_11_layer_call_and_return_conditional_losses_257058
R__inference_batch_normalization_11_layer_call_and_return_conditional_losses_256983
R__inference_batch_normalization_11_layer_call_and_return_conditional_losses_256965┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
ъ2Џ
7__inference_batch_normalization_11_layer_call_fn_257071
7__inference_batch_normalization_11_layer_call_fn_257009
7__inference_batch_normalization_11_layer_call_fn_257084
7__inference_batch_normalization_11_layer_call_fn_256996┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
з2­
I__inference_activation_14_layer_call_and_return_conditional_losses_257089б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
п2Н
.__inference_activation_14_layer_call_fn_257094б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
в2У
A__inference_add_5_layer_call_and_return_conditional_losses_257100б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
л2═
&__inference_add_5_layer_call_fn_257106б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
з2­
I__inference_activation_15_layer_call_and_return_conditional_losses_257111б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
п2Н
.__inference_activation_15_layer_call_fn_257116б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
│2░
K__inference_up_sampling2d_2_layer_call_and_return_conditional_losses_253650Я
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *@б=
;і84                                    
ў2Ћ
0__inference_up_sampling2d_2_layer_call_fn_253656Я
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *@б=
;і84                                    
ц2А
E__inference_conv2d_13_layer_call_and_return_conditional_losses_253667О
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *7б4
2і/+                            
Ѕ2є
*__inference_conv2d_13_layer_call_fn_253677О
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *7б4
2і/+                            
Р2▀
R__inference_batch_normalization_12_layer_call_and_return_conditional_losses_257177
R__inference_batch_normalization_12_layer_call_and_return_conditional_losses_257159┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
г2Е
7__inference_batch_normalization_12_layer_call_fn_257203
7__inference_batch_normalization_12_layer_call_fn_257190┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
з2­
I__inference_activation_16_layer_call_and_return_conditional_losses_257208б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
п2Н
.__inference_activation_16_layer_call_fn_257213б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ц2А
E__inference_conv2d_14_layer_call_and_return_conditional_losses_253814О
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *7б4
2і/+                            
Ѕ2є
*__inference_conv2d_14_layer_call_fn_253824О
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *7б4
2і/+                            
Р2▀
R__inference_batch_normalization_13_layer_call_and_return_conditional_losses_257274
R__inference_batch_normalization_13_layer_call_and_return_conditional_losses_257256┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
г2Е
7__inference_batch_normalization_13_layer_call_fn_257300
7__inference_batch_normalization_13_layer_call_fn_257287┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
з2­
I__inference_activation_17_layer_call_and_return_conditional_losses_257305б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
п2Н
.__inference_activation_17_layer_call_fn_257310б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
в2У
A__inference_add_6_layer_call_and_return_conditional_losses_257316б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
л2═
&__inference_add_6_layer_call_fn_257322б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ц2А
E__inference_conv2d_15_layer_call_and_return_conditional_losses_253961О
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *7б4
2і/+                            
Ѕ2є
*__inference_conv2d_15_layer_call_fn_253971О
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *7б4
2і/+                            
Р2▀
R__inference_batch_normalization_14_layer_call_and_return_conditional_losses_257365
R__inference_batch_normalization_14_layer_call_and_return_conditional_losses_257383┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
г2Е
7__inference_batch_normalization_14_layer_call_fn_257409
7__inference_batch_normalization_14_layer_call_fn_257396┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
з2­
I__inference_activation_18_layer_call_and_return_conditional_losses_257414б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
п2Н
.__inference_activation_18_layer_call_fn_257419б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
в2У
A__inference_add_7_layer_call_and_return_conditional_losses_257425б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
л2═
&__inference_add_7_layer_call_fn_257431б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
з2­
I__inference_activation_19_layer_call_and_return_conditional_losses_257436б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
п2Н
.__inference_activation_19_layer_call_fn_257441б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
│2░
K__inference_up_sampling2d_3_layer_call_and_return_conditional_losses_254110Я
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *@б=
;і84                                    
ў2Ћ
0__inference_up_sampling2d_3_layer_call_fn_254116Я
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *@б=
;і84                                    
ц2А
E__inference_conv2d_16_layer_call_and_return_conditional_losses_254127О
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *7б4
2і/+                            
Ѕ2є
*__inference_conv2d_16_layer_call_fn_254137О
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *7б4
2і/+                            
Р2▀
R__inference_batch_normalization_15_layer_call_and_return_conditional_losses_257484
R__inference_batch_normalization_15_layer_call_and_return_conditional_losses_257502┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
г2Е
7__inference_batch_normalization_15_layer_call_fn_257515
7__inference_batch_normalization_15_layer_call_fn_257528┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
з2­
I__inference_activation_20_layer_call_and_return_conditional_losses_257533б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
п2Н
.__inference_activation_20_layer_call_fn_257538б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ц2А
E__inference_conv2d_17_layer_call_and_return_conditional_losses_254274О
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *7б4
2і/+                            
Ѕ2є
*__inference_conv2d_17_layer_call_fn_254284О
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *7б4
2і/+                            
Р2▀
R__inference_batch_normalization_16_layer_call_and_return_conditional_losses_257581
R__inference_batch_normalization_16_layer_call_and_return_conditional_losses_257599┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
г2Е
7__inference_batch_normalization_16_layer_call_fn_257625
7__inference_batch_normalization_16_layer_call_fn_257612┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
з2­
I__inference_activation_21_layer_call_and_return_conditional_losses_257630б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
п2Н
.__inference_activation_21_layer_call_fn_257635б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
3B1
$__inference_signature_wrapper_255872input_3ы
!__inference__wrapped_model_253196╦H'(./01:;ABCDQRXYZ[pqwxyzЃёіІїЇџЏАбБц╣║└┴┬├╠═МнНо8б5
.б+
)і&
input_3         
ф "EфB
@
activation_21/і,
activation_21         х
I__inference_activation_12_layer_call_and_return_conditional_losses_256733h7б4
-б*
(і%
inputs          
ф "-б*
#і 
0          
џ Ї
.__inference_activation_12_layer_call_fn_256738[7б4
-б*
(і%
inputs          
ф " і          х
I__inference_activation_13_layer_call_and_return_conditional_losses_256905h7б4
-б*
(і%
inputs          
ф "-б*
#і 
0          
џ Ї
.__inference_activation_13_layer_call_fn_256910[7б4
-б*
(і%
inputs          
ф " і          х
I__inference_activation_14_layer_call_and_return_conditional_losses_257089h7б4
-б*
(і%
inputs          
ф "-б*
#і 
0          
џ Ї
.__inference_activation_14_layer_call_fn_257094[7б4
-б*
(і%
inputs          
ф " і          х
I__inference_activation_15_layer_call_and_return_conditional_losses_257111h7б4
-б*
(і%
inputs          
ф "-б*
#і 
0          
џ Ї
.__inference_activation_15_layer_call_fn_257116[7б4
-б*
(і%
inputs          
ф " і          ┌
I__inference_activation_16_layer_call_and_return_conditional_losses_257208їIбF
?б<
:і7
inputs+                            
ф "?б<
5і2
0+                            
џ ▒
.__inference_activation_16_layer_call_fn_257213IбF
?б<
:і7
inputs+                            
ф "2і/+                            ┌
I__inference_activation_17_layer_call_and_return_conditional_losses_257305їIбF
?б<
:і7
inputs+                            
ф "?б<
5і2
0+                            
џ ▒
.__inference_activation_17_layer_call_fn_257310IбF
?б<
:і7
inputs+                            
ф "2і/+                            ┌
I__inference_activation_18_layer_call_and_return_conditional_losses_257414їIбF
?б<
:і7
inputs+                            
ф "?б<
5і2
0+                            
џ ▒
.__inference_activation_18_layer_call_fn_257419IбF
?б<
:і7
inputs+                            
ф "2і/+                            ┌
I__inference_activation_19_layer_call_and_return_conditional_losses_257436їIбF
?б<
:і7
inputs+                            
ф "?б<
5і2
0+                            
џ ▒
.__inference_activation_19_layer_call_fn_257441IбF
?б<
:і7
inputs+                            
ф "2і/+                            ┌
I__inference_activation_20_layer_call_and_return_conditional_losses_257533їIбF
?б<
:і7
inputs+                            
ф "?б<
5і2
0+                            
џ ▒
.__inference_activation_20_layer_call_fn_257538IбF
?б<
:і7
inputs+                            
ф "2і/+                            ┌
I__inference_activation_21_layer_call_and_return_conditional_losses_257630їIбF
?б<
:і7
inputs+                           
ф "?б<
5і2
0+                           
џ ▒
.__inference_activation_21_layer_call_fn_257635IбF
?б<
:і7
inputs+                           
ф "2і/+                           р
A__inference_add_4_layer_call_and_return_conditional_losses_256916Џjбg
`б]
[џX
*і'
inputs/0          
*і'
inputs/1          
ф "-б*
#і 
0          
џ ╣
&__inference_add_4_layer_call_fn_256922јjбg
`б]
[џX
*і'
inputs/0          
*і'
inputs/1          
ф " і          р
A__inference_add_5_layer_call_and_return_conditional_losses_257100Џjбg
`б]
[џX
*і'
inputs/0          
*і'
inputs/1          
ф "-б*
#і 
0          
џ ╣
&__inference_add_5_layer_call_fn_257106јjбg
`б]
[џX
*і'
inputs/0          
*і'
inputs/1          
ф " і          Џ
A__inference_add_6_layer_call_and_return_conditional_losses_257316НЉбЇ
ЁбЂ
џ|
<і9
inputs/0+                            
<і9
inputs/1+                            
ф "?б<
5і2
0+                            
џ з
&__inference_add_6_layer_call_fn_257322╚ЉбЇ
ЁбЂ
џ|
<і9
inputs/0+                            
<і9
inputs/1+                            
ф "2і/+                            Џ
A__inference_add_7_layer_call_and_return_conditional_losses_257425НЉбЇ
ЁбЂ
џ|
<і9
inputs/0+                            
<і9
inputs/1+                            
ф "?б<
5і2
0+                            
џ з
&__inference_add_7_layer_call_fn_257431╚ЉбЇ
ЁбЂ
џ|
<і9
inputs/0+                            
<і9
inputs/1+                            
ф "2і/+                            ╚
R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_256781rABCD;б8
1б.
(і%
inputs          
p
ф "-б*
#і 
0          
џ ╚
R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_256799rABCD;б8
1б.
(і%
inputs          
p 
ф "-б*
#і 
0          
џ ь
R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_256856ќABCDMбJ
Cб@
:і7
inputs+                            
p
ф "?б<
5і2
0+                            
џ ь
R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_256874ќABCDMбJ
Cб@
:і7
inputs+                            
p 
ф "?б<
5і2
0+                            
џ а
7__inference_batch_normalization_10_layer_call_fn_256812eABCD;б8
1б.
(і%
inputs          
p
ф " і          а
7__inference_batch_normalization_10_layer_call_fn_256825eABCD;б8
1б.
(і%
inputs          
p 
ф " і          ┼
7__inference_batch_normalization_10_layer_call_fn_256887ЅABCDMбJ
Cб@
:і7
inputs+                            
p
ф "2і/+                            ┼
7__inference_batch_normalization_10_layer_call_fn_256900ЅABCDMбJ
Cб@
:і7
inputs+                            
p 
ф "2і/+                            ╚
R__inference_batch_normalization_11_layer_call_and_return_conditional_losses_256965rXYZ[;б8
1б.
(і%
inputs          
p
ф "-б*
#і 
0          
џ ╚
R__inference_batch_normalization_11_layer_call_and_return_conditional_losses_256983rXYZ[;б8
1б.
(і%
inputs          
p 
ф "-б*
#і 
0          
џ ь
R__inference_batch_normalization_11_layer_call_and_return_conditional_losses_257040ќXYZ[MбJ
Cб@
:і7
inputs+                            
p
ф "?б<
5і2
0+                            
џ ь
R__inference_batch_normalization_11_layer_call_and_return_conditional_losses_257058ќXYZ[MбJ
Cб@
:і7
inputs+                            
p 
ф "?б<
5і2
0+                            
џ а
7__inference_batch_normalization_11_layer_call_fn_256996eXYZ[;б8
1б.
(і%
inputs          
p
ф " і          а
7__inference_batch_normalization_11_layer_call_fn_257009eXYZ[;б8
1б.
(і%
inputs          
p 
ф " і          ┼
7__inference_batch_normalization_11_layer_call_fn_257071ЅXYZ[MбJ
Cб@
:і7
inputs+                            
p
ф "2і/+                            ┼
7__inference_batch_normalization_11_layer_call_fn_257084ЅXYZ[MбJ
Cб@
:і7
inputs+                            
p 
ф "2і/+                            ь
R__inference_batch_normalization_12_layer_call_and_return_conditional_losses_257159ќwxyzMбJ
Cб@
:і7
inputs+                            
p
ф "?б<
5і2
0+                            
џ ь
R__inference_batch_normalization_12_layer_call_and_return_conditional_losses_257177ќwxyzMбJ
Cб@
:і7
inputs+                            
p 
ф "?б<
5і2
0+                            
џ ┼
7__inference_batch_normalization_12_layer_call_fn_257190ЅwxyzMбJ
Cб@
:і7
inputs+                            
p
ф "2і/+                            ┼
7__inference_batch_normalization_12_layer_call_fn_257203ЅwxyzMбJ
Cб@
:і7
inputs+                            
p 
ф "2і/+                            ы
R__inference_batch_normalization_13_layer_call_and_return_conditional_losses_257256џіІїЇMбJ
Cб@
:і7
inputs+                            
p
ф "?б<
5і2
0+                            
џ ы
R__inference_batch_normalization_13_layer_call_and_return_conditional_losses_257274џіІїЇMбJ
Cб@
:і7
inputs+                            
p 
ф "?б<
5і2
0+                            
џ ╔
7__inference_batch_normalization_13_layer_call_fn_257287ЇіІїЇMбJ
Cб@
:і7
inputs+                            
p
ф "2і/+                            ╔
7__inference_batch_normalization_13_layer_call_fn_257300ЇіІїЇMбJ
Cб@
:і7
inputs+                            
p 
ф "2і/+                            ы
R__inference_batch_normalization_14_layer_call_and_return_conditional_losses_257365џАбБцMбJ
Cб@
:і7
inputs+                            
p
ф "?б<
5і2
0+                            
џ ы
R__inference_batch_normalization_14_layer_call_and_return_conditional_losses_257383џАбБцMбJ
Cб@
:і7
inputs+                            
p 
ф "?б<
5і2
0+                            
џ ╔
7__inference_batch_normalization_14_layer_call_fn_257396ЇАбБцMбJ
Cб@
:і7
inputs+                            
p
ф "2і/+                            ╔
7__inference_batch_normalization_14_layer_call_fn_257409ЇАбБцMбJ
Cб@
:і7
inputs+                            
p 
ф "2і/+                            ы
R__inference_batch_normalization_15_layer_call_and_return_conditional_losses_257484џ└┴┬├MбJ
Cб@
:і7
inputs+                            
p
ф "?б<
5і2
0+                            
џ ы
R__inference_batch_normalization_15_layer_call_and_return_conditional_losses_257502џ└┴┬├MбJ
Cб@
:і7
inputs+                            
p 
ф "?б<
5і2
0+                            
џ ╔
7__inference_batch_normalization_15_layer_call_fn_257515Ї└┴┬├MбJ
Cб@
:і7
inputs+                            
p
ф "2і/+                            ╔
7__inference_batch_normalization_15_layer_call_fn_257528Ї└┴┬├MбJ
Cб@
:і7
inputs+                            
p 
ф "2і/+                            ы
R__inference_batch_normalization_16_layer_call_and_return_conditional_losses_257581џМнНоMбJ
Cб@
:і7
inputs+                           
p
ф "?б<
5і2
0+                           
џ ы
R__inference_batch_normalization_16_layer_call_and_return_conditional_losses_257599џМнНоMбJ
Cб@
:і7
inputs+                           
p 
ф "?б<
5і2
0+                           
џ ╔
7__inference_batch_normalization_16_layer_call_fn_257612ЇМнНоMбJ
Cб@
:і7
inputs+                           
p
ф "2і/+                           ╔
7__inference_batch_normalization_16_layer_call_fn_257625ЇМнНоMбJ
Cб@
:і7
inputs+                           
p 
ф "2і/+                           В
Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_256609ќ./01MбJ
Cб@
:і7
inputs+                            
p
ф "?б<
5і2
0+                            
џ В
Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_256627ќ./01MбJ
Cб@
:і7
inputs+                            
p 
ф "?б<
5і2
0+                            
џ К
Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_256684r./01;б8
1б.
(і%
inputs          
p
ф "-б*
#і 
0          
џ К
Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_256702r./01;б8
1б.
(і%
inputs          
p 
ф "-б*
#і 
0          
џ ─
6__inference_batch_normalization_9_layer_call_fn_256640Ѕ./01MбJ
Cб@
:і7
inputs+                            
p
ф "2і/+                            ─
6__inference_batch_normalization_9_layer_call_fn_256653Ѕ./01MбJ
Cб@
:і7
inputs+                            
p 
ф "2і/+                            Ъ
6__inference_batch_normalization_9_layer_call_fn_256715e./01;б8
1б.
(і%
inputs          
p
ф " і          Ъ
6__inference_batch_normalization_9_layer_call_fn_256728e./01;б8
1б.
(і%
inputs          
p 
ф " і          ┌
E__inference_conv2d_10_layer_call_and_return_conditional_losses_253207љ'(IбF
?б<
:і7
inputs+                           
ф "?б<
5і2
0+                            
џ ▓
*__inference_conv2d_10_layer_call_fn_253217Ѓ'(IбF
?б<
:і7
inputs+                           
ф "2і/+                            ┌
E__inference_conv2d_11_layer_call_and_return_conditional_losses_253354љ:;IбF
?б<
:і7
inputs+                            
ф "?б<
5і2
0+                            
џ ▓
*__inference_conv2d_11_layer_call_fn_253364Ѓ:;IбF
?б<
:і7
inputs+                            
ф "2і/+                            ┌
E__inference_conv2d_12_layer_call_and_return_conditional_losses_253501љQRIбF
?б<
:і7
inputs+                            
ф "?б<
5і2
0+                            
џ ▓
*__inference_conv2d_12_layer_call_fn_253511ЃQRIбF
?б<
:і7
inputs+                            
ф "2і/+                            ┌
E__inference_conv2d_13_layer_call_and_return_conditional_losses_253667љpqIбF
?б<
:і7
inputs+                            
ф "?б<
5і2
0+                            
џ ▓
*__inference_conv2d_13_layer_call_fn_253677ЃpqIбF
?б<
:і7
inputs+                            
ф "2і/+                            ▄
E__inference_conv2d_14_layer_call_and_return_conditional_losses_253814њЃёIбF
?б<
:і7
inputs+                            
ф "?б<
5і2
0+                            
џ ┤
*__inference_conv2d_14_layer_call_fn_253824ЁЃёIбF
?б<
:і7
inputs+                            
ф "2і/+                            ▄
E__inference_conv2d_15_layer_call_and_return_conditional_losses_253961њџЏIбF
?б<
:і7
inputs+                            
ф "?б<
5і2
0+                            
џ ┤
*__inference_conv2d_15_layer_call_fn_253971ЁџЏIбF
?б<
:і7
inputs+                            
ф "2і/+                            ▄
E__inference_conv2d_16_layer_call_and_return_conditional_losses_254127њ╣║IбF
?б<
:і7
inputs+                            
ф "?б<
5і2
0+                            
џ ┤
*__inference_conv2d_16_layer_call_fn_254137Ё╣║IбF
?б<
:і7
inputs+                            
ф "2і/+                            ▄
E__inference_conv2d_17_layer_call_and_return_conditional_losses_254274њ╠═IбF
?б<
:і7
inputs+                            
ф "?б<
5і2
0+                           
џ ┤
*__inference_conv2d_17_layer_call_fn_254284Ё╠═IбF
?б<
:і7
inputs+                            
ф "2і/+                           Ћ
C__inference_model_2_layer_call_and_return_conditional_losses_255074═H'(./01:;ABCDQRXYZ[pqwxyzЃёіІїЇџЏАбБц╣║└┴┬├╠═МнНо@б=
6б3
)і&
input_3         
p

 
ф "?б<
5і2
0+                           
џ Ћ
C__inference_model_2_layer_call_and_return_conditional_losses_255206═H'(./01:;ABCDQRXYZ[pqwxyzЃёіІїЇџЏАбБц╣║└┴┬├╠═МнНо@б=
6б3
)і&
input_3         
p 

 
ф "?б<
5і2
0+                           
џ ѓ
C__inference_model_2_layer_call_and_return_conditional_losses_256170║H'(./01:;ABCDQRXYZ[pqwxyzЃёіІїЇџЏАбБц╣║└┴┬├╠═МнНо?б<
5б2
(і%
inputs         
p

 
ф "-б*
#і 
0         
џ ѓ
C__inference_model_2_layer_call_and_return_conditional_losses_256364║H'(./01:;ABCDQRXYZ[pqwxyzЃёіІїЇџЏАбБц╣║└┴┬├╠═МнНо?б<
5б2
(і%
inputs         
p 

 
ф "-б*
#і 
0         
џ ь
(__inference_model_2_layer_call_fn_255440└H'(./01:;ABCDQRXYZ[pqwxyzЃёіІїЇџЏАбБц╣║└┴┬├╠═МнНо@б=
6б3
)і&
input_3         
p

 
ф "2і/+                           ь
(__inference_model_2_layer_call_fn_255673└H'(./01:;ABCDQRXYZ[pqwxyzЃёіІїЇџЏАбБц╣║└┴┬├╠═МнНо@б=
6б3
)і&
input_3         
p 

 
ф "2і/+                           В
(__inference_model_2_layer_call_fn_256465┐H'(./01:;ABCDQRXYZ[pqwxyzЃёіІїЇџЏАбБц╣║└┴┬├╠═МнНо?б<
5б2
(і%
inputs         
p

 
ф "2і/+                           В
(__inference_model_2_layer_call_fn_256566┐H'(./01:;ABCDQRXYZ[pqwxyzЃёіІїЇџЏАбБц╣║└┴┬├╠═МнНо?б<
5б2
(і%
inputs         
p 

 
ф "2і/+                            
$__inference_signature_wrapper_255872оH'(./01:;ABCDQRXYZ[pqwxyzЃёіІїЇџЏАбБц╣║└┴┬├╠═МнНоCб@
б 
9ф6
4
input_3)і&
input_3         "EфB
@
activation_21/і,
activation_21         Ь
K__inference_up_sampling2d_2_layer_call_and_return_conditional_losses_253650ъRбO
HбE
Cі@
inputs4                                    
ф "HбE
>і;
04                                    
џ к
0__inference_up_sampling2d_2_layer_call_fn_253656ЉRбO
HбE
Cі@
inputs4                                    
ф ";і84                                    Ь
K__inference_up_sampling2d_3_layer_call_and_return_conditional_losses_254110ъRбO
HбE
Cі@
inputs4                                    
ф "HбE
>і;
04                                    
џ к
0__inference_up_sampling2d_3_layer_call_fn_254116ЉRбO
HбE
Cі@
inputs4                                    
ф ";і84                                    