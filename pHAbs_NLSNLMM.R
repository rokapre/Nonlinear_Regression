library(lme4)

Fit_NLS = function(data){
  model = nls(ALED~Amax/(1+exp((pKa-pH)/phi)),start=list(Amax=1,pKa=7,phi=0.5),data=data)
  return(model)
}

Fit_NLMM = function(data){
  startvec = c(1,7,0.5)
  names(startvec) = c("Amax","pKa","phi")
  nformall = ~ Amax / (1+exp((pKa-pH)/phi))
  nfunall = deriv(nformall,namevec=names(startvec),function.arg=c('pH','Amax','pKa','phi'))
  NLMMfull = nlmer(ALED~nfunall(pH,Amax,pKa,phi)~Amax|Trial,data=data,start=startvec)
  return(NLMMfull)
}

