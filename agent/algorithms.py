import numpy as np
import collections


class SourceDetector:

    def __init__(self, buffer_size=30, position_buffer_size=30, conc_threshold=13, grad_threshold=0.3, identify_for=10):
        # Thresholds for source detection
        self.conc_threshold = conc_threshold
        self.grad_threshold = grad_threshold

        # Buffer size
        self.buffer_size = buffer_size
        self.position_buffer_size = position_buffer_size

        # Buffers for source detection
        self.conc_buffer = collections.deque(maxlen=buffer_size)
        self.grad_x_buffer = collections.deque(maxlen=buffer_size)
        self.grad_y_buffer = collections.deque(maxlen=buffer_size)

        # Buffers for source features detection
        self.position_buffer = collections.deque(maxlen=position_buffer_size)

        # Identification steps
        self.identify_for = identify_for
        self.identification_steps = 0

    def is_source_detected(self, curr_state, position):
        # Extract values from the state
        curr_conc = curr_state[0]
        curr_grad_x = curr_state[1]
        curr_grad_y = curr_state[2]

        # Append that to the deque
        self.conc_buffer.append(curr_conc)
        self.grad_x_buffer.append(curr_grad_x)
        self.grad_y_buffer.append(curr_grad_y)
        self.position_buffer.append(position)

        # Find averages
        mean_conc = np.mean(self.conc_buffer)
        mean_grad_x = np.mean(self.grad_x_buffer)
        mean_grad_y = np.mean(self.grad_y_buffer)

        # Determine if source detected
        if (mean_conc >= self.conc_threshold and abs(mean_grad_x) <=
                self.grad_threshold and abs(mean_grad_y) <= self.grad_threshold):
            # print("Source detected")

            # Detect field parameters
            # print("Starting loc: " + str(self.position_buffer[0]))
            # print("Ending loc: " + str(self.position_buffer[-1]))

            vx = (((self.position_buffer[-1][0] - self.position_buffer[0][0]) * 0.8) /
                  (len(self.position_buffer) * 0.1))
            vy = (((self.position_buffer[-1][1] - self.position_buffer[0][1]) * 0.8) /
                  (len(self.position_buffer) * 0.1))
            k = (self.conc_buffer[-1] -
                 self.conc_buffer[0]) / (len(self.conc_buffer) * 0.1)

            # print("Adv-Diff parameters detected (vx, vy, k): " + str((vx, vy, k)))

#             print(
#                 "Diffs: " + str(self.position_buffer[-1][0] - self.position_buffer[0][0]) + ", " +
#                 str((self.position_buffer[-1][1] - self.position_buffer[0][1])))

#             print("Grad at center: " + str(curr_grad_x) + ", " + str(curr_grad_y))

            return True

        # If source not detected
        return False

    def are_source_params_identified(self, curr_state):
        self.identification_steps += 1
        self.position_buffer.append([curr_state[0], curr_state[1]])

        if self.identification_steps == self.identify_for:
            # print("Starting loc: " + str(self.position_buffer[0]))
            # print("Ending loc: " +
            #       str(self.position_buffer[-self.identify_for]))

            vy = (((self.position_buffer[-1][0] - self.position_buffer[-self.identify_for][0]) * 0.8) /
                  (self.identify_for * 0.1))
            vx = (((self.position_buffer[-1][1] - self.position_buffer[-self.identify_for][1]) * 0.8) /
                  (self.identify_for * 0.1))
            return True, (vx, vy)

        return False, None

    def reset(self):
        self.conc_buffer.clear()
        self.grad_x_buffer.clear()
        self.grad_y_buffer.clear()


class SourceParamsIdentifier:

    def __init__(self, buffer_size=10):
        self.buffer_size = buffer_size
        self.position_buffer = collections.deque(maxlen=buffer_size)

    def get_velocity_params(self):
        pass


class DestinationChooser:

    def __init__(self, frontier_distance):
        # The distance in both x and y directions where to consider the frontier to be
        self.frontier_distance = frontier_distance

    def find_destination(self, curr_location, visited, sources=[], hop_over=10, boundary_limit=5):
        # return [state[0] + self.frontier_distance, state[1]]

        # Get boundary of possible locations
        min_x = curr_location[0] - self.frontier_distance
        min_y = curr_location[1] - self.frontier_distance
        max_x = curr_location[0] + self.frontier_distance
        max_y = curr_location[1] + self.frontier_distance

        # Populate all candidates
        all_candidates = []
        i = min_x
        while i < max_x:
            all_candidates.append([i, min_y])
            all_candidates.append([i, max_y])
            i += hop_over

        j = min_y
        while j < max_y:
            all_candidates.append([min_x, j])
            all_candidates.append([max_x, j])
            j += hop_over

        # Prune candidates based on their location
        candidates = []
        for i in range(len(all_candidates)):
            candidate = all_candidates[i]
            if (candidate[0] > boundary_limit and candidate[0] < (100 - boundary_limit) and
                    candidate[1] > boundary_limit and candidate[1] < (100 - boundary_limit) and
                    visited[candidate[0], candidate[1]] == 0):
                candidates.append(candidate)

        # Get the best candidate destination, its path and farthest visited cell
        best_candidate, best_candidate_dict = self.get_best_candidate_v2(
            curr_location, candidates, visited, sources)

        # Return the best candidate and its path
        return best_candidate, best_candidate_dict["path"]

    def get_best_candidate(self, curr_location, candidates, visited):
        """
        This function returns the best candidate destination from among the `candidates` list when
        the robot is at `curr_location`. It uses the DDA algorithm to weigh the candidates based on
        how far they are from the visited regions in `visited`. Then a weighted random candidate is
        chosen. This function returns best_candidate and a dictionary containing its `weight`,
        `path` and `farthest_visited_cell`
        """
        # Initialize dictionary and weights list
        cand_dict = {}
        weights = []

        # Get centroid of the visited matrix
        (x_zeros, y_zeros) = np.where(visited == 0)
        mean_x = sum(x_zeros) / len(x_zeros)
        mean_y = sum(y_zeros) / len(y_zeros)
        unvisited_centroid = np.array([mean_x, mean_y])

        # print("Centroid of non-visited area: " + str(unvisited_centroid))

        # For each candidate, calculate the path from candidate as source and curr_location as
        # destination, compute farthest_visited_cell, and add it to dictionary
        for cand in candidates:
            path, farthest_visited_cell = self.create_path(
                cand, curr_location, visited)

            # Weight is calculated as the euclidean distance between the candidate and the farthest
            # visited cell. So the farther away the candidate is from the nearest visited cell, the
            # higher the probability of it being selected.
            vec_curr_loc_to_cand = np.array(curr_location) - np.array(cand)
            vec_curr_loc_to_centroid = np.array(
                curr_location) - unvisited_centroid

            similarity = np.dot(
                vec_curr_loc_to_cand, vec_curr_loc_to_centroid) / (np.linalg.norm(vec_curr_loc_to_cand) *
                                                                   np.linalg.norm(vec_curr_loc_to_centroid))

            weight = similarity * np.linalg.norm(
                np.array(farthest_visited_cell) - np.array(cand))

            cand_dict[tuple(cand)] = {
                "weight": weight,
                "path": path,
                "farthest_visited_cell": farthest_visited_cell
            }

            weights.append(weight)

        # Ensure that size of weights and candidates is the same
        # if len(weights) != len(candidates):
            # print("Length of weights and candidates doesn't match! Something is wrong.")

        # Convert weights into probabilities
        # TODO (Deepak): Choose the exponential power
        weights = [1.5 ** w for w in weights]
        total_weights = sum(weights)
        weight_probs = weights / total_weights

        # for cand, w in zip(candidates, weight_probs):
            # print(cand, w)

        # Get weighted random choice
        best_candidate_idx = np.random.choice(
            len(candidates), 1, p=weight_probs)

        best_candidate = candidates[best_candidate_idx[0]]

        # For now, return the best candidate and its information
        # print(f'best candidate: {best_candidate}')
        return best_candidate, cand_dict[tuple(best_candidate)]

    def get_best_candidate_v2(self, curr_location, candidates, visited, sources=[]):
        """
        This function returns the best candidate destination from among the `candidates` list when
        the robot is at `curr_location`. It uses the DDA algorithm to weigh the candidates based on
        how far they are from the visited regions in `visited`. Then a weighted random candidate is
        chosen. This function returns best_candidate and a dictionary containing its `weight`,
        `path` and `farthest_visited_cell`
        """
        # Initialize dictionary and weights list
        cand_dict = {}
        weights = []

        # Get centroid of the visited matrix
        (x_zeros, y_zeros) = np.where(visited == 0)
        mean_x = sum(x_zeros) / len(x_zeros)
        mean_y = sum(y_zeros) / len(y_zeros)
        unvisited_centroid = np.array([mean_x, mean_y])

        # print("Centroid of non-visited area: " + str(unvisited_centroid))

        # For each candidate, calculate the path from candidate as source and curr_location as
        # destination, compute farthest_visited_cell, and add it to dictionary
        for cand in candidates:
            path, farthest_visited_cell = self.create_path(
                cand, curr_location, visited)

            # Weight is calculated as the euclidean distance between the candidate and the farthest
            # visited cell. So the farther away the candidate is from the nearest visited cell, the
            # higher the probability of it being selected.
            vec_curr_loc_to_cand = np.array(curr_location) - np.array(cand)
            vec_curr_loc_to_centroid = np.array(
                curr_location) - unvisited_centroid

            similarity = np.dot(
                vec_curr_loc_to_cand, vec_curr_loc_to_centroid) / (np.linalg.norm(vec_curr_loc_to_cand) *
                                                                   np.linalg.norm(vec_curr_loc_to_centroid))
            
            # Added:
            min_dist_to_source = (np.min(np.linalg.norm(np.array(cand) - np.array(sources), axis=1))/100) * 2.5 if len(sources)>0 else 1
            # print(f"min_dist: {min_dist_to_source}")
            
            weight = similarity * np.linalg.norm(
                np.array(farthest_visited_cell) - np.array(cand)) * min_dist_to_source

            cand_dict[tuple(cand)] = {
                "weight": weight,
                "path": path,
                "farthest_visited_cell": farthest_visited_cell
            }

            weights.append(weight)

        # Ensure that size of weights and candidates is the same
        # if len(weights) != len(candidates):
            # print("Length of weights and candidates doesn't match! Something is wrong.")

        # Convert weights into probabilities
        # TODO (Deepak): Choose the exponential power
        weights = [1.5 ** w for w in weights]
        total_weights = sum(weights)
        weight_probs = weights / total_weights

        # for cand, w in zip(candidates, weight_probs):
            # print(cand, w)

        # Get weighted random choice
        best_candidate_idx = np.random.choice(
            len(candidates), 1, p=weight_probs)

        best_candidate = candidates[best_candidate_idx[0]]

        # For now, return the best candidate and its information
        # print(f'best candidate: {best_candidate}')
        return best_candidate, cand_dict[tuple(best_candidate)]

    
    def create_path(self, src, dst, visited):
        """
        This function uses the DDA algorithm to "draw" a line from `dst` (candidate) to
        `src` (curr_location) which acts as the path the formation center will take to go to
        that `dst`. At each step it also checks if that location is marked as visited in the
        visited array. The first visited location encountered is saved as `farthest_visited_cell`
        which is then returned with the entire `path` found.
        """
        path = []
        farthest_visited_cell = None

        # Case 1: When both src and dst have the same x-axis
        if src[0] == dst[0]:
            if src[1] < dst[1]:
                for y in range(src[1], dst[1]):
                    pt = [src[0], y]
                    if farthest_visited_cell == None and visited[pt[0], pt[1]] == 1:
                        farthest_visited_cell = pt
                    path.append(pt)
            else:
                for y in range(dst[1], src[1]):
                    pt = [src[0], y]
                    if farthest_visited_cell == None and visited[pt[0], pt[1]] == 1:
                        farthest_visited_cell = pt
                    path.append(pt)
            return path, farthest_visited_cell

        # Case 2: Everything else
        slope = (dst[1] - src[1]) / (dst[0] - src[0])
        if abs(slope) <= 1:
            if src[0] < dst[0]:
                for x in range(src[0], dst[0]):
                    pt = [x, int(src[1] + slope * (x - src[0]))]
                    # print("pt: " + str(pt))
                    if farthest_visited_cell == None and visited[pt[0], pt[1]] == 1:
                        farthest_visited_cell = pt
                    path.append(pt)
            else:
                for x in range(dst[0], src[0]):
                    pt = [x, int(src[1] + slope * (x - src[0]))]
                    if farthest_visited_cell == None and visited[pt[0], pt[1]] == 1:
                        farthest_visited_cell = pt
                    path.append(pt)
        else:
            if src[1] < dst[1]:
                for y in range(src[1], dst[1]):
                    pt = [int(src[0] + (y - src[1]) / slope), y]
                    if farthest_visited_cell == None and visited[pt[0], pt[1]] == 1:
                        farthest_visited_cell = pt
                    path.append(pt)
            else:
                for y in range(dst[1], src[1]):
                    pt = [int(src[0] + (y - src[1]) / slope), y]
                    if farthest_visited_cell == None and visited[pt[0], pt[1]] == 1:
                        farthest_visited_cell = pt
                    path.append(pt)
        return path, farthest_visited_cell


class Controller:

    def __init__(self, curr_location, path, destination):
        self.destination = destination
        self.curr_location_idx = 0

        # Check correct order for the path
        print("First location in path: " + str(path[0]))
        print("Last location in path: " + str(path[-1]))

        manhattan_dist_to_first = abs(
            curr_location[0] - path[0][0]) + abs(curr_location[1] - path[0][1])
        manhattan_dist_to_last = abs(
            curr_location[0] - path[-1][0]) + abs(curr_location[1] - path[-1][1])

        print("Dist to first: " + str(manhattan_dist_to_first))
        print("Dist to last: " + str(manhattan_dist_to_last))

        # Add destination as part of the path to be taken
        self.path = None
        if manhattan_dist_to_last < manhattan_dist_to_first:
            self.path = [p for p in reversed(path)]
        else:
            self.path = path

        self.path.append(destination)

    def choose_action(self, curr_location):
        """
        actions mapping: left, right, up, down --> 0, 1, 2, 3
        """

        # Default action (stay)
        action = 0
        # TODO (Deepak): Check if curr_location is close to the next location
        # if curr_location != self.path[self.curr_location_idx]:
        #     print("The state reached is unexpected! Should have been at " +
        #           str(self.path[self.curr_location_idx]) + " but is at " + str(curr_location))
        #     return 3, False
        if curr_location == self.destination:
            return 0, True

        next_location = self.path[self.curr_location_idx]
        print(f"curr:{curr_location}, next:{next_location}")

        if (next_location == curr_location):
            self.curr_location_idx += 1
            next_location = self.path[self.curr_location_idx]

        if (curr_location[0] != next_location[0] and curr_location[1] != next_location[1]):
            # Case 1: Going to next location requires taking two separate steps. This happens when
            # the next location is diagonal to the current location
            # First figure out which diagonal the next location is at
            if ((next_location[1] == curr_location[1] - 1) and
                    (next_location[0] == curr_location[0] - 1)):
                # top-left
                action = 5  # left in learning frame. Next action should be up
            elif ((next_location[1] == curr_location[1] - 1) and
                  (next_location[0] == curr_location[0] + 1)):
                # bottom-left
                action = 7  # left in learning frame. Next action should be down
            elif ((next_location[1] == curr_location[1] + 1) and
                  (next_location[0] == curr_location[0] - 1)):
                # top-right
                action = 6  # right in learning frame. Next action should be up
            elif ((next_location[1] == curr_location[1] + 1) and
                  (next_location[0] == curr_location[0] + 1)):
                # bottom-right
                action = 8  # right in learning frame. Next action should be down
        else:
            # Case 2: Next location is only one step away. This happens when next location is either
            # directly above, to left, to right, or below the curr_location
            if next_location[1] == curr_location[1] - 1:
                action = 0   # left in learning frame
            elif next_location[1] == curr_location[1] + 1:
                action = 1   # right in learning frame
            elif next_location[0] == curr_location[0] - 1:
                action = 2   # up in learning frame
            elif next_location[0] == curr_location[0] + 1:
                action = 3   # down in learning frame
            # Since the next location is reachable, increment the index
        self.curr_location_idx += 1
        # reached = False
        # if (state[0] == destination[0] and state[1] == destination[1]):
        #     reached = True
        return action, False
