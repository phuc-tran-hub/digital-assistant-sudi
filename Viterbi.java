import java.io.*;
import java.util.*;

/**
 * @author Phuc Tran
 */

public class Viterbi {

    /**
     *
     * Train a model (observation and transition probabilities) on corresponding lines (sentence and tags) from a pair
     * of training files
     *
     * @param tagsFile tag file name
     * @param sentencesFile sentence file name
     * @return A hidden Markov model Graph
     * @throws IOException
     */
    public static Map<String, Map<String, Map<String, Double>>> trainPOS(String tagsFile,
                                                                         String sentencesFile) throws IOException {

        // holds transitionTable and observationTable for easier access
        Map<String, Map<String, Map<String, Double>>> markovGraph = new TreeMap<>();

        // Opens reader and initialize transitionsTable
        BufferedReader readerTransition_1 = new BufferedReader(new FileReader(tagsFile));
        Map<String, Map<String, Double>> transitionsTable = new TreeMap<>();

        String str = "", line;

        // Create transitions table with counts
        while((line = readerTransition_1.readLine()) != null){

            // Split the line into states; lower case so "A" and "a" are the same state
            String[] states = line.toLowerCase(Locale.ROOT).split(" ");

            // Accounts for Starting State's frequencies
            if(!transitionsTable.containsKey("#")){
                Map<String, Double> startingState = new TreeMap<>();
                startingState.put(states[0], 1.0);
                transitionsTable.put("#", startingState);
            }

            else if(transitionsTable.containsKey("#")){
                Map<String, Double> startingState = transitionsTable.get("#");
                // Add new starting state if "#" key doesn't include it
                if(transitionsTable.get("#").get(states[0]) == null){
                    startingState.put(states[0], 1.0);
                    transitionsTable.put("#", startingState);
                }
                // Else, increment the value for the starting state
                else{
                    startingState.put(states[0], transitionsTable.get("#").get(states[0]) + 1);
                    transitionsTable.put("#", startingState);
                }
            }

            // Iterate through the states
            for(int i = 0; i < states.length - 1; i ++){

                // If table doesn't have the current state, add a key with the state
                // and value as a Map of the next state and 1 (first occurrence of next state)
                if(!transitionsTable.containsKey(states[i])){
                    Map<String, Double> newNextState = new TreeMap<>();
                    newNextState.put(states[i+1], 1.0);
                    transitionsTable.put(states[i], newNextState);
                    continue;
                }

                // If table has the next state for the current state, increment that next state's value
                if(transitionsTable.get(states[i]).containsKey(states[i+1])){
                    Map<String, Double> updateNextState = transitionsTable.get(states[i]);
                    updateNextState.put(states[i+1], transitionsTable.get(states[i]).get(states[i+1]) + 1);
                    transitionsTable.put(states[i], updateNextState);
                    continue;
                }

                // If the table does not have the next state, create a new Map of the next state as key and 1 (first
                // occurrence of next state) and put that Map into key current state
                if(!transitionsTable.get(states[i]).containsKey(states[i+1])){
                    Map<String, Double> newNextState = transitionsTable.get(states[i]);
                    newNextState.put(states[i+1], 1.0);
                    transitionsTable.put(states[i], newNextState);
                }
            }
        }

        // Convert counts to log probabilities
        for(Map.Entry<String, Map<String, Double>> state :transitionsTable.entrySet()) {
            // keeps track of the row counts
            int totalTransition = 0;

            // Iterate through the map to find the total row count for that state
            for (Map.Entry<String, Double> nextState : state.getValue().entrySet()) {
                totalTransition += nextState.getValue();
            }

            // Convert the counts to log probabilities (i.e. log(count/total row count))
            for(Map.Entry<String, Double> nextState : state.getValue().entrySet()) {
                Map<String, Double> updateNextState = state.getValue();
                updateNextState.put(nextState.getKey(), Math.log(nextState.getValue()/ totalTransition));
            }
        }


        readerTransition_1.close();

        // Opens new reader for observationsTable and initialize observationTable
        BufferedReader readerTransition_2 = new BufferedReader(new FileReader(tagsFile));
        BufferedReader readerObservation = new BufferedReader(new FileReader(sentencesFile));
        Map<String, Map<String, Double>> observationsTable = new TreeMap<>();

        String str_2 = "", line_2;

        // Creating observation tables with counts

        // Reading the line for tag file
        while((line = readerTransition_2.readLine()) != null){

            // Also reading the line for sentence file
            line_2 = readerObservation.readLine();

            // Remember to lowercase these states and words again to count for similar states i.e. A vs a
            String[] states = line.toLowerCase(Locale.ROOT).split(" ");
            String[] words = line_2.toLowerCase(Locale.ROOT).split(" ");

            // Iterate through the states
            for(int i = 0; i < states.length; i++){

                // If the observation table doesn't have the state, add a key with a value as a Map with
                // the matching word as key and 1.0 as value (first occurrence of word)
                if(!observationsTable.containsKey(states[i])){
                    Map<String, Double> newWord = new TreeMap<>();
                    newWord.put(words[i], 1.0);
                    observationsTable.put(states[i], newWord);
                    continue;
                }

                // If the observation table does have the word for the current state, increment the value for
                // the matching word as key
                if(observationsTable.get(states[i]).containsKey(words[i])){
                    Map<String, Double> updateWord = observationsTable.get(states[i]);
                    updateWord.put(words[i], updateWord.get(words[i]) + 1);
                    observationsTable.put(states[i], updateWord);
                    continue;
                }

                // If the observation table doesn't have the word for the current state, add a Map with
                // the matching word as key and 1.0 as value (first occurrence of word) and put that Map on
                // observationTable
                if(!observationsTable.get(states[i]).containsKey(words[i])){
                    Map<String, Double> newWord = observationsTable.get(states[i]);
                    newWord.put(words[i], 1.0);
                    observationsTable.put(states[i], newWord);
                }
            }
        }

        // Converting table to log probabilities
        for(Map.Entry<String, Map<String, Double>> state :observationsTable.entrySet()) {
            // keeps track of the row counts
            int totalObservation = 0;

            // Iterate through the map to find the total row count for that state
            for (Map.Entry<String, Double> word : state.getValue().entrySet()) {
                totalObservation += word.getValue();
            }

            // Convert the counts to log probabilities (i.e. log(count/total row count))
            for(Map.Entry<String, Double> word : state.getValue().entrySet()) {
                Map<String, Double> updateNextState = state.getValue();
                updateNextState.put(word.getKey(), Math.log(word.getValue()/ totalObservation));
            }
        }

        // Closing the readers
        readerTransition_2.close();
        readerObservation.close();
        markovGraph.put("transitions", transitionsTable);
        markovGraph.put("observations", observationsTable);
        return markovGraph;
    }

    /**
     * Perform POS Viterbi decoding and backtracking for best possible path
     *
     * @param words String that contains sequence of words
     * @param unseenWordPenalty Penalty when the state does not hold observation
     * @param markovGraph The graph to perform Viterbi decoding
     * @return A list of the best possible path for words
     */
    public static List<String> POSViterbi(String words, int unseenWordPenalty,
                                  Map<String, Map<String, Map<String, Double>>> markovGraph){

        // List of states at the current level (#); initializing #
        Set<String> currStates = new HashSet<>();
        currStates.add("#");

        // List of scores associated with the current states; initializing # = 0.0
        Map<String, Double> currScores = new TreeMap<>();
        currScores.put("#", 0.0);

        // List of Map of Key = Current State and Value = Past State, allows backtracking
        List<Map<String, String>> backTrackList = new ArrayList<>();

        // Lowercase to remove ambiguity and split into an array
        String[] observations = words.toLowerCase(Locale.ROOT).split(" ");

        // Initialize max as the lowest number possible (negative infinity)
        double max_score = Double.NEGATIVE_INFINITY;


        // Iterate from 0 to observation.length - 1
        for(int i = 0; i < observations.length; i++){

            // Create list to hold the next states and next scores
            Set<String> nextStates = new HashSet<>();
            Map<String, Double> nextScores = new HashMap<>();

            // Create the backtrack Table for current observations
            Map<String, String> backTrackTable = new TreeMap<>();

            for(String currState: currStates) {
                if (markovGraph.get("transitions").containsKey(currState)) {
                    Set<String> nextStatesSet = markovGraph.get("transitions").get(currState).keySet();
                    // Iterate through the children of the parent
                    for (String nextState : nextStatesSet) {
                        // Add children to nextStates
                        nextStates.add(nextState);

                        // Calculate nextScores = currScore + transit + obs
                        double nextScore;
                        nextScore = currScores.get(currState) + markovGraph.get("transitions").get(currState).get(nextState);

                        // If vertex does not hold the observation, -10 instead of adding the obs score
                        if (markovGraph.get("observations").containsKey(nextState) &&
                                markovGraph.get("observations").get(nextState).containsKey(observations[i])) {
                            nextScore += markovGraph.get("observations").get(nextState).get(observations[i]);
                        } else {
                            nextScore += unseenWordPenalty;
                        }

                        // If nextScores doesn't have nextState or nextScore is greater than the current nextScore for
                        // the next state, update the backTrackTable and nextScores
                        if (!nextScores.containsKey(nextState) || nextScore > nextScores.get(nextState)) {
                            nextScores.put(nextState, nextScore);
                            backTrackTable.put(nextState, currState);

                        }

                        // At the last iteration, find the max_score for backtrack
                        if (i == observations.length - 1 && nextScore > max_score) {
                            max_score = nextScore;
                        }

                    }
                }
            }

            backTrackList.add(backTrackTable);
            currStates = nextStates;
            currScores = nextScores;
        }

        // Find first nextState to backtrack by finding the state at the last observation with max_score
        String backTrack = "";
        for(String bestState: currScores.keySet()){
            if(currScores.get(bestState) == max_score){
                backTrack = bestState;
                break;
            }
        }

        // Iterate backward on the the list
        List<String> viterbiTags = new ArrayList<>();
        for(int i = backTrackList.size() - 1; i >= 0; i --){

            // Iterate through the current list of current states
            for(String stateBackTrack : backTrackList.get(i).keySet()){

                // If the backtrack state has been found, add that backtrack to the tags and update
                // backtrack state as the the previous state
                if(stateBackTrack.equals(backTrack)){
                    viterbiTags.add(0, backTrack);
                    backTrack = backTrackList.get(i).get(backTrack);
                    break;
                }
            }

        }
        return viterbiTags;
    }

    /**
     *
     * Perform POS on a test sentences file and an HMM graph
     *
     * @param filename The test sentences file
     * @param markovGraph The graph to perform Viterbi decoding
     * @throws IOException
     */
    public static void runFile(String filename, Map<String, Map<String, Map<String, Double>>> markovGraph)
            throws IOException {

        // Opens reader to start reading test sentences file
        BufferedReader reader = new BufferedReader(new FileReader(filename));
        String str = "", line;

        // Perform viterbi on each line
        while((line = reader.readLine()) != null){
            POSViterbi(line, -15, markovGraph);
        }

        reader.close();
    }

    /**
     * Perform a console-based method to give the tags from an input line
     *
     * @param markovGraph The graph to perform Viterbi decoding
     */
    public static void consoleBasedTest(Map<String, Map<String, Map<String, Double>>> markovGraph){
        System.out.println("Insert sentence to tag; Press Q to quit");
        Scanner input = new Scanner(System.in);

        // Keeps reading input until user quits with "q"
        while(input.hasNextLine()){
            if(input.next().toLowerCase(Locale.ROOT).equals("q")){
                System.out.println("Console Completed");
                break;

            }
            else {
                // Perform POS Viterbi decoding on input
                System.out.println(POSViterbi(input.nextLine(), -10, markovGraph));
            }
        }
        input.close();
    }

    /**
     * Checks how many tags are right and wrong in this POS decoding via HMM
     *
     * @param testSentencesFile The file that holds the testing sentences
     * @param testTagFile The file to check tags for testing sentences
     * @param trainSentencesFile The file to train sentences
     * @param trainTagFile The file to train tags
     * @throws IOException
     */
    public static void evaluatePerformance(String testSentencesFile, String testTagFile, String trainSentencesFile,

                                           String trainTagFile) throws IOException {

        // Initializing counts for correct and incorrect tags
        int correct = 0;
        int incorrect = 0;

        // Opens readers and calculate markovGraph
        Map<String, Map<String, Map<String, Double>>> markovGraph = trainPOS(trainTagFile, trainSentencesFile);
        BufferedReader readerSentence = new BufferedReader(new FileReader(testSentencesFile));
        BufferedReader readerTag = new BufferedReader(new FileReader(testTagFile));

        String str = "", line_1, line_2;

        // Iterate through the sentence file and tag file
        while((line_1 = readerSentence.readLine()) != null && (line_2 = readerTag.readLine()) != null){
            // Find the list of best possible path
            List<String> viterbiTags = POSViterbi(line_1, -15, markovGraph);
            String[] wordsToCheck = line_2.toLowerCase(Locale.ROOT).split(" ");

            // Only perform if the lengths of the list and array are equal
            if(viterbiTags.size() == wordsToCheck.length){
                for(int i = 0; i < viterbiTags.size(); i++){
                    if(viterbiTags.get(i).equals(wordsToCheck[i])){
                        correct += 1;
                    }
                    else{
                        incorrect += 1;
                    }
                }
            }
        }
        readerSentence.close();
        readerTag.close();

        System.out.println("The sample solution got " + correct + " tags right and " + incorrect + " wrong");
    }

    public static void main(String[] args) throws Exception {

        System.out.println("Running Brown Tests:");
        evaluatePerformance("texts/brown-test-sentences.txt", "texts/brown-test-tags.txt",
                "texts/brown-train-sentences.txt", "texts/brown-train-tags.txt");

        System.out.println("------");

        // note: added periods to simple test files
        System.out.println("Running Simple Tests:");
        evaluatePerformance("texts/simple-test-sentences.txt", "texts/simple-test-tags.txt",
                "texts/simple-train-sentences.txt", "texts/simple-train-tags.txt");

        System.out.println("------");

        System.out.println("Hard-code Strings (graph is from example:)");

        Map<String, Map<String, Map<String, Double>>> markovGraph_3 =
                trainPOS("texts/example-tags.txt", "texts/example-sentences.txt");

        System.out.println("many fish has many jobs .");
        System.out.println(POSViterbi("many fish has many jobs .", -10, markovGraph_3) + "\n");

        System.out.println("I saw a saw .");
        System.out.println(POSViterbi("I saw a saw .", -10, markovGraph_3) + "\n");

        System.out.println("I wonder what height is the book .");
        System.out.println(POSViterbi("I wonder what height is the book .", -10, markovGraph_3)
                + "\n");

        System.out.println("I am here to eat kitties .");
        System.out.println(POSViterbi("I am here to eat kitties .", -10, markovGraph_3) + "\n");

        System.out.println("-------");

        Map<String, Map<String, Map<String, Double>>> markovGraph_1 =
                trainPOS("texts/brown-train-tags.txt", "texts/brown-train-sentences.txt");
        consoleBasedTest(markovGraph_1);
    }
}
