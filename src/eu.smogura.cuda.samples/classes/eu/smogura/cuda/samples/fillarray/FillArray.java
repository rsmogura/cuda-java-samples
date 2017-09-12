/*
* The MIT License
* 
* Copyright (c) 2015-2017 Radoslaw Smogura
* 
* Permission is hereby granted, free of charge, to any person obtaining
* a copy of this software and associated documentation files (the
* "Software"), to deal in the Software without restriction, including
* without limitation the rights to use, copy, modify, merge, publish,
* distribute, sublicense, and/or sell copies of the Software, and to
* permit persons to whom the Software is furnished to do so, subject to
* the following conditions:
* 
* The above copyright notice and this permission notice shall be
* included in all copies or substantial portions of the Software.
* 
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
* NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
* LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
* OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
* WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/
package eu.smogura.cuda.samples;

import java.io.CharArrayWriter;
import javax.expression.ExpressionContext;
import javax.expression.ExpressionIntrospector;
import javax.nicl.codetranslation.naive.CudaCodeTranslator;
import javax.nicl.cuda.CudaContext;
import javax.nicl.cuda.CudaMemoryRegion;

import static javax.nicl.cuda.CUDA.*;
import javax.nicl.cuda.CudaKernel;

/**
 * Fills array with given value.
 * <p>
 * If you have problems running this code, refer to README, or if you are on OS X
 * set working directory to place where CUDA libs are (/Developer/NVIDIA/CUDA-8.0/lib)
 * or find a way to set LD_, and DYLD_LIBRARY_PATH ;)
 * </p>
 * 
 * @author Radek "Rado" Smogura
 */
public class FillArray {
    /** Size of data to operate on. */
    private static final int REGION_SIZE = 1024 * 1024 * 1024;
    
    /** How many threads should be in block, 1024 max value. */
    private static final int THREADS_PER_BLOCK = 1024;
    
    /** In how many 1-dim blocks kernel should execute */
    private static final int BLOCKS = 1024; // Ensure no reminder from division
    
    public static void main(String... args) throws Exception {
        CudaContext cudaContext = CudaContext.newContext();        
        CudaMemoryRegion dataRegion = cudaContext.allocateRegion(REGION_SIZE);
        
        System.out.println("Data before");
        Utils.printRegion(dataRegion, 32);
        
        int elements = REGION_SIZE / 4; //sizeof(int)
        int elementsPerBlock = elements / BLOCKS;
        int loopsPerThread = elementsPerBlock / THREADS_PER_BLOCK;
        int threadsCount = THREADS_PER_BLOCK; // Have to be local to be captured...
        int valueToSet = 1; 
        
        Runnable fillArrayKernel = () -> { 
            // Trick to cast to int[], as we don't want to opeate via dataRegion
            int[] data = (int[]) (Object) dataRegion; 
            
            // The starting index of data chunk
            int dataChunkBegin = blockIdx.x * elementsPerBlock + threadIdx.x;
            
            // Set values in looop
            for (int i = 0; i < loopsPerThread; i = i + 1) { // No unary
            
                data[dataChunkBegin + threadsCount * i] = valueToSet;
            }
        };

        CharArrayWriter codeBuffer = new CharArrayWriter(4096);
        ExpressionContext expressionContext = ExpressionIntrospector.expressionForFunctionalInterface(fillArrayKernel);       
        
        CudaCodeTranslator translator = CudaCodeTranslator.generateCode(expressionContext);
        translator.writeCodeToBuffer(codeBuffer);    
        
        // Install kernel. This installs only static part of it, so "kernel" object
        // can be cached and reused in subsequent calls
        CudaKernel kernel = cudaContext.installCode(
                codeBuffer.toString(), 
                translator.getKernelName(), 
                translator.getExpressionRuntimeContextFactory(), 
                translator.getCapturedArgumentMap());
        
        uint3 gridDim = new uint3(); gridDim.x = BLOCKS; gridDim.y = gridDim.z = 1;
        uint3 blockDim = new uint3(); blockDim.x = THREADS_PER_BLOCK; blockDim.y = blockDim.z = 1;
        
        // fillArrayKernel has to be passed as argument, as it's actuall instance of kernel
        // (in contast to C, where function has static nature, and doesn't carry closure)
        long kernelStart = System.nanoTime();
        kernel.execute(fillArrayKernel, gridDim, blockDim);
        long kernelEnd = System.nanoTime();
        
        System.out.format("Executed kernel in \t%d ms, \t%d ns\n",
                (kernelEnd - kernelStart) / 1_000_000, 
                kernelEnd - kernelStart);
        
        Utils.validateIntRegion(dataRegion, REGION_SIZE, valueToSet);
        Utils.printRegion(dataRegion, 32);
    }
}
