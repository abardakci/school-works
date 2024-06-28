class DecisionTreeClassifier:
    max_depht : int
    RIGHT = 1
    LEFT  = 0

    def __init__(self, max_depth: int):
        self.root = self.Leaf()
        self.root.depht = 0
        self.max_depht = max_depth

    class Leaf:
        tvalue : float
        attribute : int
        label : int

        right = None
        left = None
        depht = 0

        def __init__(self):
            self.attribute = -1
            self.label = -1
            self.tvalue = -1
            self.depht = 0
            self.right = None
            self.left = None
    def each_attribute_value(self, X:list[list[float]], attribute_number: int):
        tvalues = []

        for row in X:
            element = row[attribute_number]
            tvalues.append(element)

        return list(set(tvalues))

    def impurityEval1(self, y: list[int]):
        if not y:
            return 0

        sample_space = len(y)
        c = {}
        for val in y:
            if val not in c:
                c[val] = 1
            else:
                c[val] += 1

        impurity = 1.0
        for value in c.values():
            impurity -= (value / sample_space) ** 2

        return impurity

    def impurityEval2(self, X:list[list[float]], y:list[int], attribute: int):
        if len(y) == 0:
            return 0.0, 0.0, [], [], [], []

        tvalue = 0.0
        imp = 1.0
        threshold_candidates = self.each_attribute_value(X, attribute)

        return_left: list[int] = []
        return_right: list[int] = []
        return_Xleft: list[list[float]] = []
        return_Xright: list[list[float]] = []

        for candidate in threshold_candidates:
            left: list[int] = []
            right: list[int] = []
            Xleft: list[list[float]] = []
            Xright: list[list[float]] = []


            for index, entity in enumerate(X):
                if entity[attribute] >= candidate:
                    Xright.append(X[index])
                    right.append(y[index])
                else:
                    Xleft.append(X[index])
                    left.append(y[index])

            newimp = self.impurityEval1(left) * (len(left)/len(y)) + self.impurityEval1(right) * (len(right)/len(y))
            if newimp < imp:
                return_Xleft, return_Xright = Xleft, Xright
                return_left, return_right = left, right
                tvalue = candidate
                imp = newimp

        return imp, tvalue, return_left, return_right, return_Xleft, return_Xright

    def fit(self, X: list[list[float]], y: list[int], threshold:float):
        self.myfit(X, y, self.root, -1, threshold)

    def myfit(self, X: list[list[float]], y: list[int], predecessor: Leaf, side, threshold:float):
        p0 = self.impurityEval1(y)
        pi:float
        current_minp = 1.0

        leftlist = rightlist = list[int]
        Xleft = Xright = list[list[float]]

        calculated_threshold_value = 0.0
        chosen_attribute = -1

        for i in range (1, 5):
            pi, tvalue, newleft, newright, newXleft, newXright  = self.impurityEval2(X, y, i)
            if pi < current_minp:
                current_minp = pi

                leftlist, rightlist = newleft, newright
                Xleft, Xright = newXleft, newXright
                calculated_threshold_value = tvalue
                chosen_attribute = i


        most_common_class = max(y, key=y.count)

        newLeaf = self.Leaf()
        newLeaf.label = most_common_class
        newLeaf.attribute = chosen_attribute
        newLeaf.tvalue = calculated_threshold_value
        newLeaf.depht = predecessor.depht + 1

        if p0 - current_minp < threshold:
            if side == self.LEFT:
                predecessor.left = newLeaf
            elif side == self.RIGHT:
                predecessor.right = newLeaf
            return

        if side == -1: #Executes if first call. Here, predcessor is root
            predecessor.label = most_common_class
            predecessor.attribute = chosen_attribute
            predecessor.tvalue = calculated_threshold_value
            predecessor.depht = 0
            if predecessor.depht < self.max_depht:
                self.myfit(Xleft, leftlist, predecessor, self.LEFT, threshold)
                self.myfit(Xright, rightlist, predecessor, self.RIGHT, threshold)

        else:
            if side == self.LEFT:
                predecessor.left = newLeaf
            elif side == self.RIGHT:
                predecessor.right = newLeaf

            if newLeaf.depht < self.max_depht:
                self.myfit(Xleft, leftlist, newLeaf, self.LEFT, threshold)
                self.myfit(Xright, rightlist, newLeaf, self.RIGHT, threshold)


    def predict(self, X: list[list[float]]):
        y:list[int] = []
        for index, list in enumerate(X):
            leaf = self.root

            while leaf.left != None and leaf.right != None:
                if list[leaf.attribute] >= leaf.tvalue:
                    leaf = leaf.right
                elif list[leaf.attribute] < leaf.tvalue:
                    leaf = leaf.left
            y.append(leaf.label)

        return y