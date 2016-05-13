#!/usr/bin/env python3
""" Code that takes a CUDA program and a policy definition,
    then weave in the policies as though they were aspects
    """
__author__ = 'Bader AlBassam'
__email__ = 'bader@mail.usf.edu'

import argparse
import re
import sys

DEBUG = False

def main():
    # Parsing input parameters
    parser = argparse.ArgumentParser(description='CUDA aspect weaver.')
    parser.add_argument('input' , help="CUDA input file")
    parser.add_argument('aspects', help="Aspect input file")
    parser.add_argument('output', help="output file",
            nargs='?', default="Enforced.cu")
    parser.add_argument("-d", '--debug',  help="show debugging",
            action="store_true")
    args = parser.parse_args()

    global DEBUG
    if args.debug:
        DEBUG = True
        print("********DEBUGGING ENABLED********")

    # Getting contents of the input file here
    with open(args.input, "r") as InFile:
        inCUDA_str = InFile.read()
        InFile.seek(0)

    Policies = getPolicyList(args.aspects)
    # TODO: Put stripping logic into own function for legibility

    if  DEBUG:
        print("********STRIPPING INPUT********")
    # Strip comments out
    # This is to prevent matching the pointcuts that are actually comments
    # Stripping '//' style comments
    StrippedInput = re.sub(r'(\/\/(.*))', r'', inCUDA_str)
    # TODO: Stripping out block '/* ... */' style comments

    # Strip away unnecessary whitespace from thebeginning and end of each line
    StrippedInput = StrippedInput.splitlines()
    StrippedInput = (line.strip().lstrip() for line in StrippedInput)

    # Rejoin into a single string
    remainingCode = '\n'.join(StrippedInput)

    if  DEBUG:
        print("********STRIPPED INPUT********")
        print(str(remainingCode))
        print("********END STRIPPED INPUT********")

    enforcedCode = ""
    unEnforcedCode = ""

    # For every policy, enforce that policy
    for policy in Policies:
        # Get list of CUDA function boundaries
        # Update each time
        FunctionBoundaries = getFunctionBounds(remainingCode)

        if DEBUG:
            print("********Listing the function boundaries********")
            print(FunctionBoundaries)
        MatchingFunction = findMatchingFunction(remainingCode, FunctionBoundaries,
                policy.signature)

        if MatchingFunction is None:
            print("ERROR:\tNo functions match!")
            sys.exit()

        if DEBUG:
            print("********Matching Functions in lines: " + repr(MatchingFunction))

        # This is the "meat" of the code
        remainingCode, unEnforcedCode= extractCodeSegment(remainingCode, MatchingFunction)

        # Time to enforce our policy to the code
        enforcedCode += enforceFunction( unEnforcedCode, policy)

        if DEBUG:
            print("*" * 8 + "result of extraction for: " + repr(policy.signature))
            print(enforcedCode)

    if DEBUG:
        print("\n\n\n" + "*" * 8 + " Putting together the remaining code" + "*" * 8)
        print("*" * 8 + "Remaining")
        print(remainingCode)
        print("*" * 8 +"Enforced!")
        print(enforcedCode)

    # Writing the result into a file
    OutputCode = remainingCode + enforcedCode
    with open(args.output, "w") as OutFile:
        OutFile.write(OutputCode)

def getFunctionBounds(InputCode):
    """ Takes code and returns list of tuples represting function boundaries """
    if DEBUG:
        print('********In getFunctionBounds********')

    # http://docs.nvidia.com/cuda/cuda-c-programming-guide/#c-language-extensions
    # Set of qualifiers defining CUDA code
    Qualifiers = ('__device', '__global__', '__host__')

    # Split into list of lines
    SplitCode = InputCode.splitlines()

    # Get potential function boundary line numbers
    BodyBoundaries = []
    lineNumber = -1;

    for C in SplitCode:
        lineNumber += 1
        if C.startswith(Qualifiers):
            # Skip over prototypes to prevent false positives
            if C.endswith(';'):
                continue

            # here is the first line of a CUDA function
            # Start at this line, continue till body close
            ParenDepth = 0
            EnteredBody = False

            for bodyLineNo, BodyLine in enumerate(SplitCode[lineNumber:]):
                if not EnteredBody and  BodyLine.count('{') > 0:
                    EnteredBody = True

                # Figure out how deeply nested we are
                ParenDepth += BodyLine.count('{')
                ParenDepth -= BodyLine.count('}')

                if DEBUG:
                    print("Depth = " + repr(ParenDepth))

                # At the closing paren
                if EnteredBody and ParenDepth <= 0:
                    BodyBoundaries.append((lineNumber, lineNumber + bodyLineNo))
                    break

            if DEBUG:
                print(repr(lineNumber) +":" + C)

    if DEBUG:
        print("********Potentials are:" + repr(BodyBoundaries))
        print("********Extracted Code********")
        for i in BodyBoundaries:
            print(repr(i) + "\n" + "\n".join(SplitCode[i[0]:i[1] +1]))
        print('********End getFunctionBounds********')

    return BodyBoundaries

def findMatchingFunction(InputCode, BoundaryList, Signature):
    """ Return index of BoundaryList that points to Signature """
    # Signature of form: (Quantifier, return type, function name) 
    # Function name can be "fun" or "fun(". I make sure to check for both 
    # should return a tuple (start, end)
    qualifier, returnType, fname = Signature

    InputCode = InputCode.splitlines()
    for boundary in BoundaryList:
        # split by spaces and open parens to ease parsing
        function = re.split('\s+|\(',InputCode[boundary[0]])

        if (
                function[0] == qualifier and
                function[1] == returnType  and
                function[2] == fname
                ):
            return boundary


    return

def extractCodeSegment(InputCode, bounds):
    """ Returns tuple of (code without segment, segment) """
    start, end = bounds

    if DEBUG:
        print("********In extractCodeSegment********")

    SplitCode = InputCode.splitlines()

    PreExtract = SplitCode[:start]
    Extracted = SplitCode[start:end+1]
    PostExtract = SplitCode[end+1:]
    # Combine the unextracted segments
    UnExtracted = PreExtract + PostExtract


    if DEBUG:
        print("=" * 20)
        print("********PreExtract Code:\n" + "\n".join(PreExtract))
        print("********The Extracted Code:\n" + "\n".join(Extracted))
        print("********PostExtract Code:\n" + "\n".join(PostExtract))
        print("********Unextracted Code:\n" + "\n".join(UnExtracted))
        print("********End extractCodeSegment********")
        print("=" * 20)

    return ("\n".join(UnExtracted), "\n".join(Extracted))

def enforceFunction(InFunction, policy):
    """ Weave in the policy into function """
    # Should return the resulting enforced code

    # Enforce the  post-condition here
    InFunction = InFunction.replace("return", '{' + policy.preReturn + '}\n return')

    # Split by first occurence of a bracket as to find entry of function
    preamble, afterPrecondition = InFunction.split('{', 1)
    preamble += '{\n' + policy.preExecution

    outFunction = preamble + afterPrecondition + "\n"
    outFunction = outFunction.rsplit('}', 1)[0] + policy.postExec + '\n}\n'

    return outFunction

def getPolicyList(PolicyFile):
    """ Opens file and returns list of policies """

    if DEBUG:
        print("********Getting the Policy List********")

    with open(PolicyFile, "r") as PolFile:
        polList = [line.lstrip().rstrip() for line in PolFile]

    hasBegun = False
    inPreExec = False
    inPreReturn = False
    inPostExec = False

    # Strings to hold the code to be inserted
    preExecution = ""
    preReturn = ""
    postExecution = ""

    PolicyList = []

    for line in polList:
        if hasBegun:
            if line.startswith("@preExec"):
                inPreExec = True
                continue

            if inPreExec:
                if line.startswith("@preReturn"):
                    inPreExec = False
                    inPreReturn = True
                    continue
                if DEBUG:
                    print("PreExec:\t"+line)

                # add this line as a part of the PreExecution code
                preExecution += line + "\n"

            if inPreReturn:
                if line.startswith("@postExec"):
                    inPreReturn = False 
                    inPostExec = True
                    continue
                if DEBUG:
                    print("PreReturn\t"+line)

                # add this line as a part of the PreReturn code
                preReturn += line + "\n"

            if inPostExec:
                # exit this policies parsing here
                if line.startswith("@end"):
                    if DEBUG:
                        print("End of policy with signature\t" + repr(signature) + "\n")
                        print("Precondition code:\n" + preExecution + "\n")
                        print("Postcondition code:\n" + preReturn + "\n")
                    hasBegun = False
                    inPostExec = False
                    # Save this as a Policy object
                    PolicyList.append( Policy(signature, preExecution, preReturn, postExecution))

                    # Reset the variables for the next Policy
                    preExecution = ""
                    preReturn = ""
                    postExecution = ""
                    continue # skip the @end line

                # add this line as a part of the postExecutional code
                postExecution += line + "\n"
                
                if DEBUG:
                    print("PostExec\t" + line)

        elif line.startswith("@begin"):
            hasBegun = True
            signature = line.lstrip("@begin").split()
            if DEBUG:
                print("Got signature of:\t" + repr(signature))

    if DEBUG:
        print("********Got the Policy List********")
        for pol in PolicyList:
            print("\tNew Policy")
            print(pol.signature)
            print(pol.preExecution)
            print(pol.preReturn)
    return PolicyList

class Policy:
    """ Defines a policy in CUDA """

    def __init__(self,
            signature, preExec, preReturn, postExec):
        """ Policy Constructor"""
        self.signature = signature
        self.preExecution = preExec
        self.preReturn = preReturn
        self.postExec = postExec 
        pass

if __name__ == "__main__":
    main()
