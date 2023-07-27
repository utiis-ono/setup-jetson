//
//  ContentView.swift
//  FlowerCoreML
//
//  Created by Daniel Nugraha on 24.06.22.
//

import SwiftUI
import Foundation
import Combine
import CoreLocation
import MapKit

class UserLocationManager: NSObject, ObservableObject, CLLocationManagerDelegate {
    private let locationManager = CLLocationManager()
    private var timer: Timer?
    @Published var userLocation : CLLocationCoordinate2D?
    @Published var status: String = "You are outside the areas" // initial status message
    @Published var locationText: String = "Waiting for location..."
    @Published var center: CLLocation? // Observable center
    @Published var radius: CLLocationDistance = 0 // Observable radius

    var mapView = MKMapView() // Add this line
    var myLocationAnnotation: MKPointAnnotation? // Add this line

    override init() {
        super.init()
        self.locationManager.delegate = self
        self.locationManager.desiredAccuracy = kCLLocationAccuracyBest
        self.locationManager.requestWhenInUseAuthorization()
        self.locationManager.startUpdatingLocation()
        self.setupTimer()
    }
    private func setupTimer() {
        timer = Timer.scheduledTimer(withTimeInterval: 5, repeats: true) { [weak self] _ in
            self?.locationManager.stopUpdatingLocation()
            self?.locationManager.startUpdatingLocation()
        }
    }
    func locationManager(_ manager: CLLocationManager, didUpdateLocations locations: [CLLocation]) {
        guard let location = locations.last, let center = self.center else { return }
        self.userLocation = location.coordinate
        let distanceFromCenter = location.distance(from: center)
        if distanceFromCenter <= self.radius - 50 {
            self.status = "Inside Learning Area"
        } else if distanceFromCenter <= self.radius {
            self.status = "Inside Breakpoint Area"
        } else {
            self.status = "You are outside the areas"
        }
        self.locationText = "Latitude: \(location.coordinate.latitude), Longitude: \(location.coordinate.longitude)"
        
        // Remove old annotation
        if let oldAnnotation = self.myLocationAnnotation {
            self.mapView.removeAnnotation(oldAnnotation)
        }
        // Create new annotation for the new location
        let newAnnotation = MKPointAnnotation()
        newAnnotation.coordinate = location.coordinate
        newAnnotation.title = "My Location"
        // Add new annotation to the map
        self.mapView.addAnnotation(newAnnotation)
        // Save the new annotation
        self.myLocationAnnotation = newAnnotation
    }
}

class CircleOverlay: NSObject, MKOverlay {
    var coordinate: CLLocationCoordinate2D
    var boundingMapRect: MKMapRect
    var color: UIColor
    
    init(center: CLLocationCoordinate2D, radius: CLLocationDistance, color: UIColor) {
        self.coordinate = center
        self.color = color
        let radiusMapPoints = radius * MKMapPointsPerMeterAtLatitude(center.latitude)
        let originMapPoint = MKMapPoint(center)
        let origin = MKMapPoint(x: originMapPoint.x - radiusMapPoints, y: originMapPoint.y - radiusMapPoints)
        let size = MKMapSize(width: 2 * radiusMapPoints, height: 2 * radiusMapPoints)
        self.boundingMapRect = MKMapRect(origin: origin, size: size)
    }
}

struct MapView: UIViewRepresentable {
    @ObservedObject var userLocationManager = UserLocationManager()
    var center: CLLocationCoordinate2D
    var radius: CLLocationDistance
    
    func makeUIView(context: Context) -> MKMapView {
        let mapView = MKMapView()
        mapView.delegate = context.coordinator
        let regionRadius = radius * 2.0  // make the region larger
        let region = MKCoordinateRegion(center: center, latitudinalMeters: regionRadius, longitudinalMeters: regionRadius)
        mapView.setRegion(region, animated: true)
        mapView.showsUserLocation = true  // Here: Display the user's location by default.
        return mapView
    }

    func updateUIView(_ uiView: MKMapView, context: Context) {
        uiView.removeOverlays(uiView.overlays)
        uiView.removeAnnotations(uiView.annotations)  // Remove all annotations
        let overlay = CircleOverlay(center: center, radius: radius, color: UIColor.blue)
        uiView.addOverlay(overlay)
        // Add user annotation if it is available
        if let userLocation = userLocationManager.userLocation {
            let userAnnotation = MKPointAnnotation()
            userAnnotation.coordinate = userLocation
            userAnnotation.title = "My Location" // 追加
            uiView.addAnnotation(userAnnotation)
        }
        // 半径250mの円を追加
        let overlaySecondCircle = CircleOverlay(center: center, radius: radius-50, color: UIColor.red)
        uiView.addOverlay(overlaySecondCircle)
    }

    func makeCoordinator() -> Coordinator {
        return Coordinator()
    }
    
    class Coordinator: NSObject, MKMapViewDelegate {
        func mapView(_ mapView: MKMapView, rendererFor overlay: MKOverlay) -> MKOverlayRenderer {
            if let circleOverlay = overlay as? CircleOverlay {
                let circleRenderer = MKCircleRenderer(overlay: circleOverlay)
                circleRenderer.fillColor = circleOverlay.color.withAlphaComponent(0.2)
                circleRenderer.strokeColor = circleOverlay.color
                circleRenderer.lineWidth = 1
                return circleRenderer
            }
            return MKOverlayRenderer()
        }
        //In your mapView(_:viewFor:) method
        func mapView(_ mapView: MKMapView, viewFor annotation: MKAnnotation) -> MKAnnotationView? {
            if annotation is MKUserLocation {
//                let pin = MKAnnotationView(annotation: annotation, reuseIdentifier: nil)
//                let size = CGSize(width: 20, height: 20)
//                pin.image = UIImage(named: "flower-icon")?.resized(to: size)
//                pin.displayPriority = .defaultHigh
//                return pin
                return nil
            }
            return nil
        }
    }
}

func loadLastDataItems(url: URL, itemCount: Int) -> String {
    do {
        let text = try String(contentsOf: url, encoding: .utf8)
        let items = text.split(separator: ",")
        let lastItems = items.suffix(itemCount)
        return lastItems.joined(separator: "\n")
    } catch {
        print("Error: \(error)")
        return ""
    }
}

struct CircleOutline: Shape {
    var progress: CGFloat
    
    func path(in rect: CGRect) -> Path {
        let startAngle = Angle(degrees: -90)
        let endAngle = Angle(degrees: -90 + Double(progress * 360))
        var path = Path()
        path.addArc(center: CGPoint(x: rect.midX, y: rect.midY),
                    radius: rect.width / 2,
                    startAngle: startAngle,
                    endAngle: endAngle,
                    clockwise: false)
        return path
    }
}
//位置情報をお花にする
extension UIImage {
    func resized(to size: CGSize) -> UIImage {
        UIGraphicsBeginImageContextWithOptions(size, false, 0.0)
        self.draw(in: CGRect(origin: .zero, size: size))
        let newImage = UIGraphicsGetImageFromCurrentImageContext()
        UIGraphicsEndImageContext()
        return newImage ?? self
    }
}
struct ContentView: View {
    @ObservedObject var model = FLiOSModel()
    @ObservedObject var userLocationManager = UserLocationManager()
    @State var preparedExport = false
    @State private var startTime: Date? = nil
    @State private var timerAnimationProgress: CGFloat = 0.0
    @State private var timerText: String = "00:00"
    @State private var orangeCirclesProgress: [CGFloat] = Array(repeating: 0, count: 8)
    @State private var completeTextOffset: CGFloat = 0
    @State private var distanceInMeters: Double = 0.0  // 状態変数として定義
    
    var numberFormatter: NumberFormatter = {
        var nf = NumberFormatter()
        nf.usesGroupingSeparator = false
        nf.numberStyle = .none
        return nf
    }()
    
    let timer = Timer.publish(every: 1, on: .main, in: .common).autoconnect()
    
    private func repeatOrangeCirclesAnimation() {
        for i in 0..<8 {
            DispatchQueue.main.asyncAfter(deadline: .now() + Double(i)) {
                withAnimation(.linear(duration: 1.0)) {
                    orangeCirclesProgress[i] = 1.0
                }
            }
        }
        DispatchQueue.main.asyncAfter(deadline: .now() + 8) {
            for i in 0..<8 {
                orangeCirclesProgress[i] = 0.0
            }
            if model.federatedServerStatus != .completed(info: "") {
                repeatOrangeCirclesAnimation()
            }
        }
    }
    private func startTimerAnimation(duration: TimeInterval) {
        withAnimation(.linear(duration: duration)) {
            timerAnimationProgress = 1.0
        }
    }
    
    private func updateTimeText(elapsedTime: TimeInterval) {
        let minutes = Int(elapsedTime) / 60
        let seconds = Int(elapsedTime) % 60
        let fraction = Int((elapsedTime.truncatingRemainder(dividingBy: 1)) * 100)
        timerText = String(format: "%02d:%02d.%02d", minutes, seconds, fraction)
    }
    
    var body: some View {
        GeometryReader { geometry in
            VStack(spacing: 0) {
                //Spacer().frame(height: 50)
                Text("Flower iOS")
                    .font(.largeTitle)
                Spacer()
                MapView(center: CLLocationCoordinate2D(latitude: model.serverLatitude, longitude: model.serverLongitude), radius: model.radius)
                //MapView(center: CLLocationCoordinate2D(latitude: 35.9518709, longitude: 139.6544998), radius: 300)
                    .frame(height: geometry.size.width/2)
                    Text(userLocationManager.status)
                        .font(.system(size: 20))
                    Text(userLocationManager.locationText)
                        .font(.system(size: 10))
                Form {
                    Section(header: Text("Scenario")) {
                        HStack{
                            Picker("Select a Scenario", selection: $model.scenarioSelection) {
                                ForEach(model.scenarios, id: \.self) {
                                    Text($0.description)
                                }
                            }
                        }
                    }
                    Section(header: Text("Prepare Dataset")) {
                        HStack {
                            Text("Training Dataset: \(self.model.trainingBatchStatus.description)")
                            Spacer()
                            Button(action: {
                                model.prepareTrainDataset()
                            }) {
                                switch self.model.trainingBatchStatus {
                                case .notPrepared:
                                    Text("Start")
                                case .ready:
                                    Image(systemName: "checkmark")
                                default:
                                    ProgressView()
                                }
                            }
                        }
                        HStack {
                            Text("Test Dataset: \(self.model.testBatchStatus.description)")
                            Spacer()
                            Button(action: {
                                model.prepareTestDataset()
                            }) {
                                switch self.model.testBatchStatus {
                                case .notPrepared:
                                    Text("Start")
                                case .ready:
                                    Image(systemName: "checkmark")
                                default:
                                    ProgressView()
                                }
                            }
                            .disabled(model.trainingBatchStatus != Constants.PreparationStatus.ready)
                        }
                    }
                    
                    Section(header: Text("Federated Learning")) {
                        HStack {
                            Text("Prepare Federated Client")
                            Spacer()
                            Button(action: {
                                model.initMLFlwrClient()
                            }) {
                                switch model.mlFlwrClientStatus {
                                case .notPrepared:
                                    Text("Start")
                                case .ready:
                                    Image(systemName: "checkmark")
                                default:
                                    ProgressView()
                                }
                            }
                            .disabled(model.testBatchStatus != .ready || model.trainingBatchStatus != .ready)
                        }
                        HStack {
                            Text("Server Hostname: ")
                            TextField("Server Hostname", text: $model.hostname)
                                .multilineTextAlignment(.trailing)
                        }
                        HStack {
                            Text("Server Port: ")
                            TextField( "Server Port", value: $model.port, formatter: numberFormatter)
                                .multilineTextAlignment(.trailing)
                        }
                        HStack {
                            if model.federatedServerStatus == .ongoing(info: "") {
                                Button(action: {
                                    model.abortFederatedLearning()
                                }) {
                                    Text("Stop").foregroundColor(.red)
                                }
                            }
                            Spacer()
                            Button(action: {
                                let duration: TimeInterval = 10.0 // Set the desired duration in seconds
                                startTimerAnimation(duration: duration)
                                model.startFederatedLearning()
                                startTime = Date() // Add this line to set the startTime
                                if model.federatedServerStatus != .completed(info: "") {
                                    repeatOrangeCirclesAnimation()
                                }
                            }) {
                                switch model.federatedServerStatus {
                                case .idle:
                                    Text("Start")
                                case .completed:
                                    Text("Rerun FL")
                                    
                                default:
                                    ProgressView()
                                }
                            }
                            .disabled(model.mlFlwrClientStatus != .ready)
                        }
                        HStack{
                            Button(action: {
                                model.benchmarkSuite.exportBenchmark()
                            }) {
                                Text("Reload Log Text")
                            }
                        }
                        HStack{
                            ZStack {
                                Circle()
                                    .fill(Color.yellow)
                                    .frame(width: geometry.size.width * 1 / 3, height: geometry.size.width * 1 / 3)
                                
                                ForEach(0..<8) { i in
                                    Circle()
                                        .fill(Color.orange)
                                        .frame(width: 80, height: 80)
                                        .scaleEffect(orangeCirclesProgress[i])
                                        .opacity(orangeCirclesProgress[i])
                                        .offset(x: 0, y: -100)
                                        .rotationEffect(.degrees(Double(i) * 360 / 8))
                                }
                                
                                Text(timerText)
                                    .font(.system(size: 28))  // フォントサイズを変更する場合、.system(size: フォントサイズ)を指定します
                                    .foregroundColor(.black)  // 文字の色を変更する場合、.foregroundColor(.色)を指定します
                                    .onReceive(Timer.publish(every: 0.1, on: .main, in: .common).autoconnect(), perform: { _ in
                                        if model.federatedServerStatus == .ongoing(info: "") {
                                            if let startTime = startTime {
                                                let elapsedTime = Date().timeIntervalSince(startTime)
                                                updateTimeText(elapsedTime: elapsedTime)
                                            }
                                        }
                                    })
                            }
                            ScrollView {
                                Text(loadLastDataItems(url: model.benchmarkSuite.getBenchmarkFileUrl(), itemCount: 100))
                                    .font(.system(size: 10))
                            }
                            .frame(height: geometry.size.width/2)
                        }
                    }
                    
                    Section(header: Text("Benchmark")) {
                        HStack{
                            Text("Prepare Benchmark Export")
                            Spacer()
                            Button(action: {
                                model.benchmarkSuite.exportBenchmark()
                                preparedExport = true
                            }) {
                                Text("Start").disabled(preparedExport)
                            }
                        }
                        if model.benchmarkSuite.benchmarkExists() || preparedExport {
                            ShareLink(item:model.benchmarkSuite.getBenchmarkFileUrl())
                        }
                    }
                    
                    Section(header: Text("Local Training")) {
                        HStack {
                            Text("Prepare Local Client")
                            Spacer()
                            Button(action: {
                                model.initLocalClient()
                            }) {
                                switch model.localClientStatus {
                                case .notPrepared:
                                    Text("Start")
                                case .ready:
                                    Image(systemName: "checkmark")
                                default:
                                    ProgressView()
                                }
                            }
                            .disabled(model.testBatchStatus != .ready || model.trainingBatchStatus != .ready)
                        }
                        Stepper(value: $model.epoch, in: 1...10, label: { Text("Epoch: \(model.epoch)")})
                            .disabled(model.localClientStatus != .ready)
                        HStack {
                            switch model.localTrainingStatus {
                            case .idle:
                                Text("Local Train")
                            default:
                                Text(model.localTrainingStatus.description)
                            }
                            Spacer()
                            Button(action: {
                                model.startLocalTrain()
                            }) {
                                switch model.localTrainingStatus {
                                case .idle:
                                    Text("Start")
                                case .completed:
                                    Image(systemName: "checkmark")
                                default:
                                    ProgressView()
                                }
                            }
                            .disabled(model.localClientStatus != .ready)
                        }
                        HStack {
                            switch model.localTestStatus {
                            case .idle:
                                Text("Local Test")
                            default:
                                Text(model.localTestStatus.description)
                            }
                            Spacer()
                            Button(action: {
                                model.startLocalTest()
                            }) {
                                switch model.localTestStatus {
                                case .idle:
                                    Text("Start")
                                case .completed:
                                    Image(systemName: "checkmark")
                                default:
                                    ProgressView()
                                }
                            }
                            .disabled(model.localTrainingStatus != .completed(info: ""))
                        }
                    }
                    
                }
            }
            .background(Color(UIColor.systemGray6))
            .onAppear {
                userLocationManager.center = CLLocation(latitude: model.serverLatitude, longitude: model.serverLongitude) // set center when ContentView appears
                userLocationManager.radius = model.radius // set radius when ContentView appears
            }
        }
    }
}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}
//private func getColorForPercentage(_ percentage: Float) -> Color {
//    if percentage >= 0 && percentage < 50 {
//        return Color.blue
//    } else if percentage >= 50 && percentage < 70 {
//        return Color.green
//    } else if percentage >= 70 && percentage < 90 {
//        return Color.orange
//    } else {
//        return Color.red
//    }
//}
//                    Section(header: Text("Device Performance")) {
//                        HStack {
//                            VStack {
//                                Text("CPU Usage: ")
//                                Text("\(cpuUsageValue, specifier: "%.2f") %")
//                                    .foregroundColor(getColorForPercentage(cpuUsageValue))
//                            }
//                            Spacer()
//                            VStack {
//                                Text("Memory Usage: ")
//                                Text("\(memoryUsageValue, specifier: "%.2f") %")
//                                    .foregroundColor(getColorForPercentage(memoryUsageValue))
//                            }
//                        }
//                    }
//                    .onReceive(Timer.publish(every: 1, on: .main, in: .common).autoconnect(), perform: { _ in
//                        cpuUsageValue = cpuUsage()
//                        memoryUsageValue = memoryUsage()
//                        print("CPU Usage: \(cpuUsageValue)") // デバッグ用の出力
//                        print("Memory Usage: \(memoryUsageValue)") // デバッグ用の出力
//                        cpuMemoryData.addDataPoint(cpu: cpuUsageValue, memory: memoryUsageValue)
//                    })
// GPUとRAMのグラフ表示だが重くてアプリが落ちる
//                    .onReceive(Timer.publish(every: 1, on: .main, in: .common).autoconnect(), perform: { _ in
//                        cpuUsageValue = cpuUsage()
//                        memoryUsageValue = memoryUsage()
//                        print("CPU Usage: \(cpuUsageValue)") // デバッグ用の出力
//                        print("Memory Usage: \(memoryUsageValue)") // デバッグ用の出力
//                        cpuMemoryData.addDataPoint(cpu: cpuUsageValue, memory: memoryUsageValue)
//                    })
//                    VStack {
//                        HStack {
//                            LineView(data: cpuMemoryData.cpuData.map { Double($0) }, title: "CPU [%]")
//                                            .padding()
//                                            .frame(width: geometry.size.width / 2 - 20, height: geometry.size.height*0.5) // この行を変更
//
//                            LineView(data: cpuMemoryData.memoryData.map { Double($0) }, title: "RAM [%]")
//                                            .padding()
//                                            .frame(width: geometry.size.width / 2 - 20, height: geometry.size.height*0.5) // この行を変更
//                        }
//                    }
//func cpuUsage() -> Float {
//    var cpuLoad = host_cpu_load_info()
//    var cpuLoadInfoCount = UInt32(MemoryLayout.size(ofValue: cpuLoad) / MemoryLayout<integer_t>.size)
//
//    let result = withUnsafeMutableBytes(of: &cpuLoad) { ptr in
//        return host_statistics(mach_host_self(), HOST_CPU_LOAD_INFO, ptr.bindMemory(to: Int32.self).baseAddress!, &cpuLoadInfoCount)
//    }
//
//    if result != KERN_SUCCESS {
//        return -1
//    }
//
//    let totalTicks = cpuLoad.cpu_ticks.0 + cpuLoad.cpu_ticks.1 + cpuLoad.cpu_ticks.2 + cpuLoad.cpu_ticks.3
//    let idleTicks = cpuLoad.cpu_ticks.0
//    let usage = Float(totalTicks - idleTicks) / Float(totalTicks)
//    return usage * 100
//}
//
//func memoryUsage() -> Float {
//    var taskInfo = mach_task_basic_info()
//    var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size) / 4
//
//    let kerr = withUnsafeMutableBytes(of: &taskInfo) { ptr in
//        return task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), ptr.bindMemory(to: Int32.self).baseAddress!, &count)
//    }
//
//    if kerr == KERN_SUCCESS {
//        let usedMemory = Float(taskInfo.resident_size) / (1024.0 * 1024.0)
//        let totalMemory = Float(ProcessInfo.processInfo.physicalMemory) / (1024.0 * 1024.0)
//        let usage = (usedMemory / totalMemory) * 100
//        return usage
//    } else {
//        return -1
//    }
//}

//*ここにあったコード
