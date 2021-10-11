# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        return successorGameState.getScore()

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """
    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.
          Here are some method calls that might be useful when implementing minimax.
          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1
          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action
          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        print("MinimaxAgent with depth ", self.depth)
        legal = gameState.getLegalActions(0)
        successors = [gameState.generateSuccessor(0, action) for action in legal]
        maxValue = -float('inf')
        goalIndex = 0
        for x in range(len(successors)):
            actionValue = self.value(successors[x], 1, 0)
            if actionValue > maxValue:
                maxValue = actionValue
                goalIndex = x

        return legal[goalIndex]

    def MAXvalue(self, gameState, agentIndex, depthSoFar):
        legal = gameState.getLegalActions(agentIndex)
        successors = [gameState.generateSuccessor(agentIndex, action) for action in legal]
        x = -float('inf')
        for successor in successors:
            x = max(x, self.value(successor, 1, depthSoFar))
        return x

    def MINvalue(self, gameState, agentIndex, depthSoFar):
        legal = gameState.getLegalActions(agentIndex)
        successors = [gameState.generateSuccessor(agentIndex, action) for action in legal]
        x = float('inf')
        for successor in successors:
            if agentIndex + 1 == gameState.getNumAgents():  # all the ghost(s) finished their turn, Pacman next
                x = min(x, self.value(successor, 0, depthSoFar + 1))
            else:                                           # Another ghost's turn
                x = min(x, self.value(successor, agentIndex + 1, depthSoFar))
        return x

    def value(self, gameState, agentIndex, depthSoFar):

        "If requisite no. of searches complete, evaluation function"
        if depthSoFar == self.depth or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        "If agentIndex is 0, perform MAX"
        if agentIndex == 0:
            return self.MAXvalue(gameState, agentIndex, depthSoFar)
        "Else (if agentindex > 0), perform MIN"
        if agentIndex > 0:
            return self.MINvalue(gameState, agentIndex, depthSoFar)

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        def MAXvalue(gameState, agentIndex, depth, alpha, beta):
            depthSoFar = depth + 1
            legal = gameState.getLegalActions(agentIndex)

            # check for terminal states
            if gameState.isWin() or gameState.isLose() or depthSoFar == self.depth:
                return self.evaluationFunction(gameState)
            x = -float('inf')

            for action in legal:
                successor = gameState.generateSuccessor(0, action)
                x = max(x, MINvalue(successor, 1, depthSoFar, alpha, beta))
                # if x is greater than beta return x
                if x >= beta:
                    return x
                # alpha is the max of min nodes
                alpha = max(alpha, x)
            return x


        def MINvalue(gameState, agentIndex, depthSoFar, alpha, beta):
            legal = gameState.getLegalActions(agentIndex)
            x = float('inf')

            if gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)

            for action in legal:
                successor = gameState.generateSuccessor(agentIndex, action)
                if agentIndex == (gameState.getNumAgents() - 1):
                    x = min(x, MAXvalue(successor, 0, depthSoFar + 1, alpha, beta))

                else:
                    x = min(x, MAXvalue(successor, agentIndex + 1, depthSoFar, alpha, beta))
                    if x <= alpha:
                        return x
                    beta = min(beta, x)
            return x

        # Alpha-Beta Pruning
        legal = gameState.getLegalActions(0)
        currentScore = 0
        returnAction = ''
        alpha = -float('inf')
        beta = float('inf')

        for action in legal:
            successor = gameState.generateSuccessor(0, action)
            score = MINvalue(successor, 0, 1, alpha, beta)

            if score > currentScore:
                returnAction = action
                currentScore = score

            # prune beta from tree
            if score > beta:
                return returnAction
            alpha = max(alpha, score)
        return returnAction

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"

        def MAXvalue(gameState, depthSoFar):
            legalMoves = gameState.getLegalActions(0)
            if len(legalMoves) == 0 or gameState.isWin() or gameState.isLose() or depthSoFar == self.depth:
                return self.evaluationFunction(gameState), None

            x = -(float("inf"))
            Act = None

            for action in legalMoves:
                successor = EXPvalue(gameState.generateSuccessor(0, action), 1, depthSoFar)
                sVal = successor[0]
                if (x < sVal):
                    x, Act = sVal, action
                return (x, Act)

        def EXPvalue(gameState, agentIndex, depthSoFar):
            legalMoves = gameState.getLegalActions(agentIndex)
            if len(legalMoves) == 0:
                return self.evaluationFunction(gameState), None

            for action in legalMoves:
                if (agentIndex == gameState.getNumAgents() - 1):
                    successor = MAXvalue(gameState.generateSuccessor(agentIndex, action), depthSoFar + 1)
                else:
                    successor = EXPvalue(gameState.generateSuccessor(agentIndex, action), agentIndex + 1, depthSoFar)
                sVal = successor[0]
                prob = sVal / len(legalMoves)
                return (prob, action)
        max = MAXvalue(gameState, 0)
        return max

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
