import XCTest
@testable import ShapedArray

final class ShapedArrayTests: XCTestCase {
    func testExample() throws {
        // This is an example of a functional test case.
        // Use XCTAssert and related functions to verify your tests produce the correct
        // results.
        print(ShapedArray(shape: [4], scalars: [1, 2, 3, 4]))
    }
}