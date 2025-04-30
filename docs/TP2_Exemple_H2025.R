library(quantmod)
library(TTR)
library(xts)  #required as the data was saved as an xts object
library(caret)
library(AUC)
library(PerformanceAnalytics)
library(tidyr)
library(ggplot2)
library(dplyr)
library(vip)  #Toolbox pour l'importance des variables
library(rattle) #Interface pour afficher les arbres de décision

# Load data
# Vous pouvez utiliser des listes pour les titres à downloader
# Apple: AAPL
# Alphabet: GOOG
# Tesla: TSLA
# Microsoft: MSFT (pas utilisé pour le TP2: Je l'ai utilisé pour valider les procédures)
# VXN: Indice de volatilité implicite des titres du NASDAQ
stocklist <- c("AAPL","GOOG","TSLA","MSFT","^VXN")
getSymbols(stocklist)
# Indicateurs provenant de la base de données FRED
# La base FRED contient la plupart des données économiques aux USA et est rendue disponible
# par la réserve fédérale de St-Louis
# Écarts de crédits corporatifs: BAMLH0A0HYM2
# Pente entre les bons du Trésor 6 mois et les Fed Funds: T6MFF
Fredlist <- c("BAMLH0A0HYM2","T6MFF")
getSymbols(Fredlist,src='FRED')

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Pour l'exemple, on utilise le titre de Apple

# Vous pouvez transformer la périodicité des vos données
# La paquet XTS est utilisé pour gérer les séries chronologiques.
# Il permet de transformer, aggréger, choisir des prédiodes en utilisant les dates.
# Lorsque vous downloadez des séries en utilisant le paquet quantmod, ces séries sont présentées automatiquement
# dans le format xts
#----------------------------------------
# Exemple d'application xts sur les séries chronologiques
# Vous pouvez aggréger les données sur la semaine
AAPL_WVol<- apply.weekly(Vo(AAPL),sum) # Somme du volume quotidien
# Vous pouvez convertir vos données weekly avec la dernière journée de transaction
AAPL_W <- apply.weekly(AAPL,last)
#-----------------------------------------
# Les graphiques
# On peut facilement présenter un graphique des titres choisis avec le paquet quantmod
# Lorsqu'on développe un modèle, il est important de bien comprendre le ou les titres
# J'aime bien débuter (comme la plupart des analystes) par une approche visuelle
chartSeries(AAPL,
            type="line",
            theme=chartTheme('white'))

# Vous pouvez utiliser seulement une année ou une période spécifique
# Ici, on fait un graphique du cours depuis 2022 jusqu'à maintenant
chartSeries(AAPL,
            type="line",
            subset='2022/',
            theme=chartTheme('white'))


# Vous pouvez utiliser les bâtons ou chandelles
# sur une courte période
chartSeries(AAPL,
            type="bar",
            subset='2023-11::2023-12',
            theme=chartTheme('white'))
# sur une plus longue période
chartSeries(AAPL,
            type="candlesticks",
            subset='2007-03',
            theme=chartTheme('white'))


#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Tous les indicateurs techniques sont sur les prix des titres sauf les indicateurs du VXN, Credit et Yield Curve.
# Pour vous faciliter la vie et la mienne, je les ajoute au dataframe des différents titres.

AAPL.plus = cbind(AAPL,VXN$VXN.Adjusted,BAMLH0A0HYM2,T6MFF)
AAPL.plus <- AAPL.plus[complete.cases(AAPL.plus),]
colnames(AAPL.plus) <- c("Open","High","Low","Close","Volume","Adjusted","VXN","Credit","YC") 

# Pour la suite de l'analyse on a besoin des prix
# Étant donné que l'analyse est à court terme, on utilise les prix d'ouverture et de fermeture et le non return index
# NB: On ne peut acheter au close et donc on trouve la position au close mais pn transige le lendemain à l'ouverture
# Vous créez 3 variables
# Rendement sur 1 jour: return_1d (Utilisé dans l'analyse technique pour mieux comprendre les indicateurs)
# Rendement sur 5 jours: return_5d
# Indicateur de prix : Dir - Variable binaire qui indique si le prix du titre dans 5 jours est supérieur au cours actuel

dir = ifelse(AAPL.plus$Close >= lag(AAPL.plus$Open, 4), 1, 0) 
colnames(dir) <- c("dir")
return_1d <- (AAPL.plus$Close - AAPL.plus$Open)/AAPL.plus$Open
colnames(return_1d) <- c("return_1d")
return_5d <- (AAPL.plus$Close - lag(AAPL.plus$Open,4))/lag(AAPL.plus$Open,4)
colnames(return_5d) <- c("return_5d")
# Vous ajoutez ces variables aux informations sur votre titre
Stock_info <- cbind(AAPL.plus,dir,return_1d,return_5d)
# Pour simplifier le TP2, vous ne conserver que les lignes (ou journées) où toutes les informations sont disponibles
# NB: En pratique, on ne fait pas ça
Stock_info <- Stock_info[complete.cases(Stock_info),]

# Valeurs pertinentes
dir <- Stock_info$dir
price <- Stock_info$Close
high <- Stock_info$High
low <- Stock_info$Low
vol <- Stock_info$Volume
return_1d <- Stock_info$return_1d
return_5d <- Stock_info$return_5d
vol_index <- Stock_info$VXN
credit <- Stock_info$Credit
YC <- Stock_info$YC

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# En premier, vous testez chacun des indicateurs techniques et vous développez des règles de trading
# Bien entendu, un des particularités des ces règles est que vous êtes ''in sample'' et donc que vous connaissez l'avenir
# Néanmoins, il importe en apprentissage supervisé de bien connaître les facteurs utilisés

# Pour le TP2, on utilise des indicateurs techniques. Étant donné qu'ils sont basés sur l'information de prix et de volume
# des titres, ça facilite le travail pour le TP2. En pratique, les traders vont utiliser plusieurs autres indicateurs.

# Vous le constaterez, lorsqu'on est ''in sample'' est qu'on fixe des seuils, on peut toujours trouver une règle de
# trading qui sera rentable. En pratique, ce n'est pas si simple. Les études montrent que peu d'indicateurs techniques
# génèrent des rendements excédentaires.

# Pour chacun des indicateurs, je vous suggère de consulter la documentation du paquet TTR et ses références.

# RSI

# Les RSI est un des indicateurs techniques les plus populaires. Les valeurs de base sont:
# - En bas de 30: Le titre est sur-vendu
# - En haut de 70: Le titre est sur-acheté
# L'indice est calculé sur un nombre de jours, souvent 14

day=14 # Valeur usuelle pour le RSI de 14 jours 
signal <- c()                    #initialisez un vecteur pour inclure le signal
rsi <- RSI(price, day)     #Fonction pour le RSI du paquet TTR

# Charting with Trading rule
chartSeries(AAPL,
            type = 'line',
            subset="2009-08::2025-09-15",
            theme=chartTheme('white'))
addRSI(n = day, maType = "EMA", wilder = TRUE)

# Signal sur 1 jour

signal [1:day] <- 0            #Assigne le signal à 0, soit pas de trade jusqu'à day+1. On pourrait laisser les NA si vous préférez

# J'ai seulement mis un seuil à 40 pour avoir une position longue < 40 ou neutre si > 40

for (i in (day+1): length(price)){
  if (rsi[i] < 40){             #buy if rsi < 40
    signal[i] <- 1
  }else {                       #no trade all if rsi > 40
    signal[i] <- 0
  }
}
signal<-reclass(signal,price) # la fonction reclass met l'objet selon le format xts

trade <- Lag(signal,1) # Si on fait l'analyse sur 1 periode. 
ret<-return_1d * trade
names(ret) <- 'RSI'
charts.PerformanceSummary(ret) # Graphique de la performance utilisant le paquet Performance Analytics

# Signal sur 5 jours
trade <- Lag(signal,5) # Si on fait l'analyse sur 5 periodes
ret<-return_5d * trade
names(ret) <- 'RSI Holding 5 days'
charts.PerformanceSummary(ret)


# Bollinger Band
# Un des indicateurs les plus utilisés. On présente le titre en fonction de l'écart-type de ses rendements
# On utilise souvent 2 écart-types (valeur commune en statistique)
# Le titre est supposé être sur-acheté ou sur-vendu si son prix se retrouve au-delà ou en deçà de 2 écart-types
# On utilise une fenêre mobile de 20 jours (à peu près 1 mois) pour calculer l'écart-type

day=20
bb <-BBands(price,n=day, sd=2)
tail(bb,n=5)

# Charting with Trading rule
chartSeries(AAPL,
            type = 'line',
            subset="2009-08::2025-09-15",
            theme=chartTheme('white'))
addBBands(n=20,sd=2)

# Pour transformer le BBand en signal, on utilise une valeur standardisée
# Signal (price - dn)(up-dn) = Scale Bollinger Band

signal = rep(0,length(price))

# NB: J'ai mis une position short lorsque le titre se retrouve à plus de 2 écart-types. C'est vraiment très élevé
# NB: Lorsque le titre se retrouve en bas du seuil de 2 écart-types, j'ai mis une position neutre car cela peut signifier
# un problème majeur pour la firme qui ne sera pas incorporé dans l'indice
# NB: Les seuils sont mis de façon non-scientifique. Vous pouvez les changer et améliorer la performance

for (i in (day+1): length(price)){
  signal[i]=lag(signal[i],1) # Keep the same position
  if (bb$pctB[i]>1){             # Overvalued
    signal[i] <- -1
  }else if (bb$pctB[i]<0) {    # Deep undervalue - Attention!
    signal[i]= 0
  }else {           #Inside Band
    signal[i]=1
  }
}
signal<-reclass(signal,price)

trade <- Lag(signal,1) # Si on fait l'analyse sur 1 periode
ret<-return_1d * trade
names(ret) <- 'Bollinger Band'
charts.PerformanceSummary(ret)

# Signal sur 5 jours
trade <- Lag(signal,5) # Si on fait l'analyse sur 5 périodes ( 1 semaine)
ret<-return_5d * trade
names(ret) <- 'Bollinger Band 5 days'
charts.PerformanceSummary(ret)

# Chaikin Money Flows
#Développé par Marc Chaikin, Chaikin Money Flow mesure le montant d'argent entrant dans un actif sur une période spécifique.
# Le volume des flux monétaires constitue la base de la ligne de distribution d’accumulation.
# L'indicateur additionne le volume de flux monétaire pour une période rétrospective spécifique, généralement 20 ou 21 jours. 

cmf <- CMF(Stock_info[,c("High","Low","Close")], Stock_info[,"Volume"])
tail(cmf)

# Charting with Trading rule
chartSeries(price,
            type = 'line',
            subset="2009-08::2025-09-15",
            theme=chartTheme('white'))
addTA(CMF(Stock_info[,c("High","Low","Close")], Stock_info[,"Volume"]))


signal = rep(0,length(price))
for (i in (day+1): length(price)){
  if (cmf[i]>0.20){             #
    signal[i] <- 1
  }else if (cmf[i]< -0.20) {    
    signal[i]=-1
  }else {
    signal[i]=0
  }
}

signal<-reclass(signal,price)

# Signal sur 1 jour
trade <- Lag(signal,1) # Si on fait l'analyse sur 1 journée
ret<-return_1d * trade
names(ret) <- 'Chaikin Money Flow'
charts.PerformanceSummary(ret)

# Signal sur 5 jours
trade <- Lag(signal,5) # Si on fait l'analyse sur 5 jours
ret<-return_5d * trade
names(ret) <- 'Chaikin Money Flow 5 days'
charts.PerformanceSummary(ret)


# Lane Stochastic Oscillator

# Indicateur qui identifie les titres sur-achetés ou sur-vendus. On utilise les hauts,bas des 14 deniers jours. 
# 


stochOSC <- WPR(Stock_info[,c("High","Low","Close")])
tail(stochOSC)

# Charting with Trading rule
chartSeries(price,
            type = 'line',
            subset="2009-08::2025-09-15",
            theme=chartTheme('white'))
addTA(WPR(Stock_info[,c("High","Low","Close")]))

# Lorsque l'indicateur surpasse 90, le prix du titre est à des sommets.
# Si l'indicateur est en base de 50, le titre est plus bas que son sommet mais encore positit. Toutefois, un indicateur
# près de 0 est indicatif d'un creux pour le titre.

signal = rep(0,length(price))
for (i in (day+1): length(price)){
  if (stochOSC[i]>0.90){             # Overbought
    signal[i] <- -1
  }else if (stochOSC[i]< 0.50) {    
    signal[i]=1
  }else {
    signal[i]=0
  }
}
signal<-reclass(signal,price)

# Signal sur 1 jour
trade <- Lag(signal,1) # Si on fait l'analyse sur 1 jour
ret<-return_1d * trade
names(ret) <- 'Lane Oscillator'
charts.PerformanceSummary(ret)

# Signal sur 5 jours
trade <- Lag(signal,5) # Si on fait l'analyse sur 5 jours
ret<-return_5d * trade
names(ret) <- 'Lane Oscillator 5 days'
charts.PerformanceSummary(ret)


# Trend Detection Index
# Nouvel indicateur qui tente d'identifier les tendances de prix pour un titre.
# On utilise deux indicateurs
# DI: Directional Index - Momentum sur les prix sur d jours
# TDI: on regarde si le momentum s'est accélé lorsqu'on DI par rapport à une période plus longue 
# (multiple=2 impique un période de day 2 fois plus longue)

day=20
tdi = TDI(price, n = day, multiple = 2)
tail(tdi)

# Charting with Trading rule
chartSeries(price,
            type = 'line',
            subset="2009-08::2025-09-15",
            theme=chartTheme('white'))
addTA(TDI(price, n = 20, multiple = 2))

day = 40 # le trend ajoute une perte additionnelle de 20 périodes
signal = rep(0,length(price))
for (i in (day+1): length(price)){
  if (tdi$tdi[i]>0 & tdi$di[i]>0){             # Momentum
    signal[i] <- 1
  }else if (tdi$tdi[i]>0 & tdi$di[i]<0) {    # Lost momentum
    signal[i]=-1
  }else {
    signal[i]=0
  }
}
signal_tdi<-reclass(signal,price)


# Signal sur 1 jour
trade <- Lag(signal,1) # Si on fait l'analyse sur 1 jour
ret<-return_1d * trade
names(ret) <- 'Trend Direction Index'
charts.PerformanceSummary(ret)

# Signal sur 5 jours
trade <- Lag(signal,5) # Si on fait l'analyse sur 5 jours
ret<-return_5d * trade
names(ret) <- 'Trend Direction Index 5 days'
charts.PerformanceSummary(ret)


# Momentum 1 days
# Compare le prix actuel à son prix il y a 1 jour
# Achète le titre si son prix est supérieur au prix du jour précédent
# Forme la plus simple de l'analyse technique
# En général si on prend en compte les frais de transaction, cette stratégie est peu efficace

day=1
M <- momentum(price, n=day)
tail(M)

# Charting with Trading rule
chartSeries(price,
            type = 'line',
            subset="2009-08::2025-09-15",
            theme=chartTheme('white'))
addTA(momentum(price, n=day))

signal = rep(0,length(price))
for (i in (day+1): length(price)){
  if (M[i]>0){             
    signal[i] <- 1
  }else {
    signal[i]=0
  }
}
signal<-reclass(signal,price)

# Signal sur 1 jour
trade <- Lag(signal,1) # Si on fait l'analyse sur 1 jour
ret<-return_1d * trade
names(ret) <- 'Trend Direction Index'
charts.PerformanceSummary(ret)

# Signal sur 5 jours
trade <- Lag(signal,5) # Si on fait l'analyse sur 5 jours. Ici, on conserve le titre pour une période de 5 jours
ret<-return_5d * trade
names(ret) <- 'Trend Direction Index 5 days'
charts.PerformanceSummary(ret)


# Momentum 5 days
# Compare le prix actuel à son prix il y a 5 jours
# Idem au momentum 1 jour

day=5
M <- momentum(price, n=day)
tail(M)

# Charting with Trading rule
chartSeries(price,
            type = 'line',
            subset="2009-08::2025-09-15",
            theme=chartTheme('white'))
addTA(momentum(price, n=day))

signal = rep(0,length(price))
for (i in (day+1): length(price)){
  if (M[i]>0){             
    signal[i] <- 1
  }else {
    signal[i]=0
  }
}
signal<-reclass(signal,price)

# Signal sur 1 jour
trade <- Lag(signal,1) # Si on fait l'analyse sur 1 jour
ret<-return_1d * trade
names(ret) <- 'Trend Direction Index'
charts.PerformanceSummary(ret)

# Signal sur 5 jours
trade <- Lag(signal,5) # Si on fait l'analyse sur 5 jours
ret<-return_5d * trade
names(ret) <- 'Trend Direction Index 5 days'
charts.PerformanceSummary(ret)


#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Trading VXN
# VXN est la volatilité implicite des options sur le NASDAQ
# La règle est simple: lorqu'il y a panique sur les marchés financiers, la volatilité explose
# Il s'agit souvent d'une opportunité pour entrer dans le marché
# Pour faciliter sa modélisation, j'ai utilisé un RSI

# RSI - Volatility
#L'idée de base est d'être dans le marché soit dans une période de panique ou dans une période d'optimisme

day=14
signal <- c()                    #initialize vector
VXN_rsi <- RSI(vol_index, day)       #rsi with lags of day
signal [1:day+1] <- 0            #0 because no signal until day+1

for (i in (day+1): length(price)){
  if (VXN_rsi[i] < 30 || VXN_rsi[i] >80 ){      #buy if rsi < 20 or rsi>80
    signal[i] <- 1
  }else {                       #no trade all if rsi > 30
    signal[i] <- 0
  }
}

signal<-reclass(signal,price)

# Signal sur 1 jour
trade <- Lag(signal,1) # Si on fait l'analyse sur 1 jour
ret<-return_1d * trade
names(ret) <- 'RSI on VXN'
charts.PerformanceSummary(ret)

# Signal sur 5 jour
trade <- Lag(signal,5) # Si on fait l'analyse sur 5 jours
ret<-return_5d * trade
names(ret) <- 'RSI on VXN 5 days'
charts.PerformanceSummary(ret)


# Credit signal
# Lorsque les écarts de crédit augmentent, ce n'est jamais bon signe.
# À l'inverse loesque les écarts de crédit baissent, cela signifie que les investisseurs sont ppositifs pour l'avenir et souhaitent prendre plus de risque

# RSI - Credit
day=14
signal <- c()                    #initialize vector
Cr_rsi <- RSI(credit)               #rsi with lags of day
signal [1:day+1] <- 0            #0 because no signal until day+1

for (i in (day+1):length(price)){
  if (Cr_rsi[i] < 50){             #buy if rsi < 50
    signal[i] <- 1
  }else {                       #no trade all if rsi > 50
    signal[i] <- 0
  }
}

signal<-reclass(signal,price)

# Signal sur 1 jour
trade <- Lag(signal,1) # Si on fait l'analyse sur 1 jour
ret<-return_1d * trade
names(ret) <- 'RSI on Credit'
charts.PerformanceSummary(ret)

# Signal sur 5 jours
trade <- Lag(signal,5) # Si on fait l'analyse sur 5 jours
ret<-return_5d * trade
names(ret) <- 'RSI on Credit 5 days'
charts.PerformanceSummary(ret)


# Yield curve - Fed Expectations
# Lorsque les marchés s'attendent à un resserrement monétaire, la bas de courbe de taux d'intérêt a tendance à devenir
# plus pentu. Une mesure est l'écart entre les taux d'intérêt 6 mois et les taux sur les fonds fédéraux.
# On modélise le signal en utilisant un RSI

# RSI - Yield Curve
day=14
signal <- c()                    #initialize vector
YC_rsi <- RSI(YC)                #rsi with lags of day
signal [1:day+1] <- 0            #0 because no signal until day+1

for (i in (day+1): length(price)){
  if (YC_rsi[i] < 40){             #buy if rsi < 30
    signal[i] <- 1
  }else {                       #no trade all if rsi > 30
    signal[i] <- 0
  }
}

signal<-reclass(signal,price)

# Signal sur 1 jour
trade <- Lag(signal,1) # Si on fait l'analyse sur 1 jour
ret<-return_1d * trade
names(ret) <- 'RSI on YC'
charts.PerformanceSummary(ret)

# Signal sur 5 jours
trade <- Lag(signal,5) # Si on fait l'analyse sur 5 jours
ret<-return_5d * trade
names(ret) <- 'RSI on YC 5 days'
charts.PerformanceSummary(ret)


#------------------------------------------------------------------------------------------

# Avant d'estimer votre modèle, dans un apprentissage supervisé, il importe de comprendre ses données

# Vous regroupez les indicateurs techniques bruts et non les signaux car les seuils seront trouvés par l'apprentissage
# Vous retardez les indicateurs de 5 jours

AAPL.c =AAPL$AAPL.Close
AAPL.o = AAPL$AAPL.Open
# SMA
sma5 = lag(SMA(AAPL.c, n = 5))  #notice the use of the lag function to take lagged values
# EMA
ema5 = lag(EMA(AAPL.c, n = 5))
# MACD
macd1 = lag(MACD(AAPL.c),5)
# RSI
rsi1 = lag(RSI(AAPL.c), 5)
# BBands
bband_1=lag(BBands(AAPL.c,n=20, sd=2)$pctB,5)
# Chaikin Money Flows
cmf_1 <- lag(CMF(AAPL[,c("AAPL.High","AAPL.Low","AAPL.Close")], AAPL[,"AAPL.Volume"]),5)
# Lane Oscillator
stochOSC_1 <- lag(WPR(AAPL[,c("AAPL.High","AAPL.Low","AAPL.Close")]),5)
# log returns en %
ret1 = 100*lag((AAPL.c-lag(AAPL.c,1))/ lag(AAPL.c,1),5) # Momentum 1 jour
ret5 = 100*lag((AAPL.c-lag(AAPL.c,5))/ lag(AAPL.c,5),5) # Momentum 5 jours
# VXN
rsi_VXN = lag(RSI(vol_index), 5)
# Credit RSI
rsi_Credit = lag(RSI(credit),5)
# Yield Curve RSI
rsi_YC = lag(RSI(YC), 5)

# price direction indicator
dir = ifelse(AAPL.c >= lag(AAPL.o, 4), 1, 0)  #direction variable compared to  5 day before price
# Rendement sur 4 jours de l'Open au Close
Stock_perf <- (AAPL.c/lag(AAPL.o, 4))-1

# Combine les signaux pour entraîner les modèles

d_ex1 = merge.xts(Stock_perf,AAPL.c,dir,macd1,ret5,rsi1,bband_1,cmf_1,stochOSC_1,rsi_VXN,rsi_Credit,rsi_YC)
# change column names
colnames(d_ex1) = c("Return","Prix","Direction","MACD","signal","Ret1","Ret5","RSI","BBand","CMF","LSO","VXN","Credit","YC")
# remove NAs
d_ex1 = na.omit(d_ex1)

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#+ On regarde graphiquement les données, le signal et les indicateurs techniques avant de faire l'estimation

# Graphique d'Apple avec le signal
chartSeries(d_ex1[,2],theme = "white", name = "Apple Prix de fermeture et Indicateur de direction", subset='2022/')
addTA(d_ex1[, 3], col = 1, legend = "Direction")  #Direction

# Graphique qui montre les différents facteurs
# On crée un dataset avec le format long
# convert to dataframe
temp = subset(d_ex1, select = -c(Return,Prix) ) # Enlève le rendement et le prix
d_plot = data.frame(Date = index(temp), coredata(temp))
d_plot_long = pivot_longer(d_plot, -c(Date, Direction), values_to = "value",
                           names_to = "Indicator")

# change direction pour un format facteur
d_plot_long$Direction = as.factor(d_plot_long$Direction)

(p2_ex = ggplot(d_plot_long, aes(Date, value, color = Indicator)) + geom_path(stat = "identity") +
    facet_grid(Indicator ~ ., scale = "free") + theme_minimal())

# Graphique montrant Directon en fonction de l'indicateur technique

p2_ex = ggplot(d_plot_long, aes(value, Indicator, fill = Direction)) +
  geom_boxplot()

p2_ex + theme_minimal() + labs(title = "Indicateurs techniques vs Direction des prix") +
  scale_fill_manual(name = "Price Direction", values = c("orange", "lightblue"))

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# Préparation des données en ensemble d'entraînement et de test

# convert to data frame
d_ex2 = as.data.frame(d_ex1)
# convert direction to a factor for classification
d_ex2$Direction = as.factor(d_ex2$Direction)
idx1 = c(1:round(nrow(d_ex2) * 0.8))  #create index for first 80% values to be in the testing set
d_train = d_ex2[idx1, ]  #training set
d_test = d_ex2[-idx1, ]  #testing set

# Si vous voulez calculer la performance de la stratégie
d_perf_train <- d_train[,1]
d_perf_test <- d_test[,1]

# Vous enlevez les données de rendement et de prix car on n'en a pas besoin pour la modélisation
d_train = subset(d_train, select = -c(Return,Prix) ) # Enlève le rendement et le prix
d_test = subset(d_test, select = -c(Return,Prix) ) # Enlève le rendement et le prix

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# Modèle Logit

set.seed(999) # Permet de toujours utiliser le même ensemble de simulation
# control
cntrl1 = trainControl(method = "timeslice", initialWindow = 250, horizon = 30,
                      fixedWindow = TRUE)
# preprocesing
prep1 = c("center", "scale")
# logistic regression
logit_ex1 = train(Direction ~ ., data = d_train, method = "glm", family = "binomial",
                  trControl = cntrl1, preProcess = prep1)
logit_ex1  #final model accuracy

summary(logit_ex1$finalModel)  #summary of the final model

# Imprime l'importance des variables
vip(logit_ex1, geom = "point") + theme_minimal()

# Prévision avec le modèle Logit sur le modèle Test

pred_Logit = predict(logit_ex1, newdata = d_test)  #prediction on the test data

# Matrice de confusion

confusionMatrix(data = pred_Logit, reference = d_test$Direction)

# Plot AUC
plot(roc(pred_Logit,d_test$Direction))

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#+
# Arbres de classification

# On fait une validation croisée avec 10 paquets et on bootstrap 3 fois
ctrl <- trainControl(method = "repeatedcv",
                     number = 10,
                     repeats = 3)


cart_01 <- train(Direction~., data = d_train, method = "rpart",preProcess = prep1,trControl = ctrl)
summary(cart_01)

# L'arbre de décision
fancyRpartPlot(cart_01$finalModel, sub = NULL)

# L'importance des variables
vip(cart_01, geom = "point") + theme_minimal()

# Prévision sur l'échantillon test
pred_Tree = predict(cart_01, newdata = d_test)  #prediction on the test data

confusionMatrix(data = pred_Tree, reference = d_test$Direction)

# Plot AUC
plot(roc(pred_Tree,d_test$Direction))

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Random Forest


rf_01 <- train(Direction~., data = d_train, method = "rf",metric="Accuracy")

summary(rf_01)

# L'importance des variables
vip(rf_01, geom = "point") + theme_minimal()


# Prévision sur l'échantillon test
pred_Rf = predict(rf_01, newdata = d_test)  #prediction on the test data

confusionMatrix(data = pred_Rf, reference = d_test$Direction)

# Plot AUC
plot(roc(pred_Rf,d_test$Direction))

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Gradient Boosting

gbm_grid = expand.grid(interaction.depth = c(1, 2, 3),
                       n.trees = 5000,
                       shrinkage = c(0.05,0.1, 0.3),
                       n.minobsinnode = 20)
head(gbm_grid)

set.seed(42)
gbm_01 = train(Direction ~ .,data = d_train,
               trControl = trainControl(method = "cv", number = 5),
               method = "gbm",tuneGrid = gbm_grid,verbose = FALSE)

summary(gbm_01)

# Prévision sur l'échantillon test
pred_GBM = predict(gbm_01, newdata = d_test)  #prediction on the test data

confusionMatrix(data = pred_GBM, reference = d_test$Direction)

# Plot AUC
plot(roc(pred_GBM,d_test$Direction))


#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

