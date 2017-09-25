import os

from aslib_scenario.aslib_scenario import ASlibScenario

class ASlibScenarioDL(ASlibScenario):
    
    def find_files(self):
        '''
            find all expected files in self.dir_
            fills self.found_files
        '''
        expected = ["description.txt", "algorithm_runs.arff", "cv.arff"]

        for expected_file in expected:
            full_path = os.path.join(self.dir_, expected_file)
            if not os.path.isfile(full_path):
                self.logger.error("Required file not found: %s" % (full_path))
                sys.exit(2)
            else:
                self.found_files.append(full_path)

