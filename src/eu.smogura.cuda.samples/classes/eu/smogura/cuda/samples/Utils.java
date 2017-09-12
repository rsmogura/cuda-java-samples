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

import javax.nicl.cuda.CudaMemoryRegion;

/**
 *
 * @author Radek "Rado" Smogura
 */
public class Utils {
    public static void validateIntRegion(CudaMemoryRegion dataRegion, int size, int expectedValue) throws IllegalAccessException {
        for (int i = 0; i < size; i+=4) {
            int value = dataRegion.getInt(i);
            if (value != expectedValue)
                throw new AssertionError(String.format("At position %d, expected %d, but was %d", i, expectedValue, value));
        }
    }
    
    public static void printRegion(CudaMemoryRegion dataRegion, int elements) throws Exception {
        for (int i = 0; i < elements; i++) {
            if (i % 8 == 0)
                System.out.format("\n%03d: ", i);
            System.out.format("0x%04X ", dataRegion.getInt(i * 4));
        }
        System.out.println();
    }
}
