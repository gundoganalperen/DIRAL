from collections import defaultdict

class Vehicle:
    """
    Each vehicle has position and and neighboring vehicle table.
    """
    # Vehicle simulator: include all the information for a vehicle

    def __init__(self, num_users, id, pos_x, pos_y, start_direction, velocity):
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.pos = [self.pos_x, self.pos_y]
        self.direction = start_direction
        self.velocity = velocity
        self.id = id
        # This variable includes tuple list wit
        #self.pos_of_neighrbors = {}
        #self.pos_of_neighrbors_transmitted = {}  # update the table after we take the step i.e. send the transmission.
        self.max_tracked_users = num_users  # this variable will be used to inialize the observations list
        self.pos_of_neighbors = defaultdict(dict)
        self.pos_of_neighbors_transmitted = defaultdict(dict)
        self.initialize_tables()

    def initialize_tables(self):
        """
        Initialize the observation tables
        :return:
        """
        for user in range(self.max_tracked_users):
            self.pos_of_neighbors[user]["xpos"] = 0
            self.pos_of_neighbors[user]["ypos"] = 0
            self.pos_of_neighbors[user]["seq_number"] = 0
            self.pos_of_neighbors[user]["last_updated"] = 0

    def received_update(self, piggybacked_positions):
        """
        Updates its table based on the received transmission
        :param piggybacked_positions: nested dictionary includes the positions of vehicles.
        :return:
        """
        for user in piggybacked_positions.keys():
            if piggybacked_positions[user]["seq_number"] > self.pos_of_neighbors[user]["seq_number"]:
                # Update the position with the piggybacked message.
                self.pos_of_neighbors[user]["xpos"] = piggybacked_positions[user]["xpos"]
                self.pos_of_neighbors[user]["ypos"] = piggybacked_positions[user]["ypos"]
                self.pos_of_neighbors[user]["seq_number"] = piggybacked_positions[user]["seq_number"]
                self.pos_of_neighbors[user]["last_updated"] = 0  # reset this number.

    def get_piggybacked_positions(self):
        """
        returns the transmitted position.
        :return:
        """
        return self.pos_of_neighbors_transmitted

    def periodic_update(self):
        """
        Update the tables every time we transferred a message.
        :return:
        """
        self.pos_of_neighbors_transmitted = self.pos_of_neighbors
        self.pos_of_neighbors[self.id]["seq_number"] += 1 # every ue updates is own seq number.
        self.pos_of_neighbors[self.id]["xpos"] = self.pos_x # every ue updates is own seq number.
        self.pos_of_neighbors[self.id]["ypos"] = self.pos_y # every ue updates is own seq number.
        for user in range(self.max_tracked_users):
            if user == self.id:
                self.pos_of_neighbors[self.id]["last_updated"] = 0  # since we updated this number
                continue
            else:
                self.pos_of_neighbors[user]["last_updated"] += 1

    def get_x_pos(self):
        """
        Return the x coordinate of the user.
        :return:
        """
        return self.pos_x
