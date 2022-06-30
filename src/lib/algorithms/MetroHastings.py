from dataclasses import dataclass


class MetroHastings:

        

        def __init__(self, data, sampleFreq, burnIn, seed, state, temperature, networkList, likelihoodList):
                self.data = data
                self.freq = sampleFreq
                self.burn = burnIn
                self.seed = seed
                self.state = state
                self.logLikelihood = self.state.calculateLikelihood()
                self.temp = temperature
                self.logPrior = self.state.calculatePrior()
                self.logPost = self.logLikelihood + self.logPrior
                self.likelihoodList = likelihoodList
                self.networkList = networkList
        

        def run(self):
                

                for i in range(self.freq):
                        accept = False
                        logHastings = self.state.propose() ##
                        op = self.state.getOperation().getName() ##

                        if logHastings != Utils.INVALID_MOVE: ##

                                if Utils.DEBUG_MODE and not self.state.isValidState():
                                        print("INVALID state after operation and validation!!!"
                                        + op + "\n" + self.state.toString() + "\n" + self.state.getNetwork())
                                

                                logPriorNext = self.state.calculatePrior()
                                logLikelihoodNext = self.state.calculateLikelihood();
                                logNext = logLikelihoodNext + logPriorNext

                                logAlpha = (logNext - self.logPost) / (self.temp + logHastings)

                                self.state.getOperation().optimize(logAlpha)

                                if logAlpha >= Math.log(Randomizer.getRandomDouble()):
                                        self.logLikelihood = logLikelihoodNext
                                        self.logPrior = logPriorNext
                                        self.logPost = logNext
                                        accept = True
                                        self.state.accept(logAlpha)
                                else:
                                        self.state.undo(logAlpha)
                                        if(SNAPPLikelihood.useApproximateBayesian):
                                                self.logLikelihood = self.state.calculateLikelihood()
                                                self.logPost = self.logPrior + self.logLikelihood
                        else:       
                                self.state.undo(Utils.INVALID_MOVE)



                        if (Utils.DEBUG_MODE and not self.state.isValidState()):
                                print("INVALID state!!!"
                                        + op + "\n" + self.state.toString() + "\n" + self.state.getNetwork())
                

                return
