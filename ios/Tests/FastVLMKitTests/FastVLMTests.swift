import XCTest
@testable import FastVLMKit

@available(iOS 17.0, macOS 14.0, *)
final class FastVLMTests: XCTestCase {
    
    func testFastVLMInitialization() throws {
        // This test will need a real model file to work
        // For now, we test that the class exists and compiles
        XCTAssertNotNil(FastVLM.self)
    }
    
    func testAnswerPlaceholder() async throws {
        // Placeholder test - requires actual model file
        // Will be implemented once model conversion is working
        
        // let vlm = try FastVLM(modelPath: "test_model.mlpackage")
        // let answer = try await vlm.answer(image: testImage, question: "What is this?")
        // XCTAssertFalse(answer.isEmpty)
    }
}