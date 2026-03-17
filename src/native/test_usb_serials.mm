/*
 * test_usb_serials.mm
 *
 * Enumerate all USB devices via IOKit and report whether each one has
 * a real iSerialNumber programmed in its firmware.
 *
 * Build:
 *   clang -framework IOKit -framework Foundation test_usb_serials.mm \
 *         -o test_usb_serials
 *
 * Run:
 *   ./test_usb_serials
 */

#import <Foundation/Foundation.h>
#import <IOKit/IOKitLib.h>
#import <IOKit/usb/IOUSBLib.h>
#include <stdio.h>
#include <string.h>

int main(void) {
    @autoreleasepool {

    CFMutableDictionaryRef matching = IOServiceMatching(kIOUSBDeviceClassName);
    if (!matching) {
        fprintf(stderr, "IOServiceMatching failed\n");
        return 1;
    }

    io_iterator_t iterator;
    kern_return_t kr = IOServiceGetMatchingServices(kIOMainPortDefault, matching, &iterator);
    if (kr != KERN_SUCCESS) {
        fprintf(stderr, "IOServiceGetMatchingServices failed: 0x%x\n", kr);
        return 1;
    }

    printf("\n%-8s %-8s %-12s %-32s %s\n",
           "VID", "PID", "LocationID", "Product Name", "iSerialNumber");
    printf("%-8s %-8s %-12s %-32s %s\n",
           "-------", "-------", "-----------", "-------------------------------", "-------------");

    io_service_t service;
    int total = 0, have_serial = 0;

    while ((service = IOIteratorNext(iterator))) {
        int vid = 0, pid = 0;
        uint32_t loc = 0;

        CFNumberRef vidRef = (CFNumberRef)IORegistryEntryCreateCFProperty(
            service, CFSTR("idVendor"), kCFAllocatorDefault, 0);
        CFNumberRef pidRef = (CFNumberRef)IORegistryEntryCreateCFProperty(
            service, CFSTR("idProduct"), kCFAllocatorDefault, 0);
        CFNumberRef locRef = (CFNumberRef)IORegistryEntryCreateCFProperty(
            service, CFSTR("locationID"), kCFAllocatorDefault, 0);
        CFStringRef nameRef = (CFStringRef)IORegistryEntryCreateCFProperty(
            service, CFSTR("USB Product Name"), kCFAllocatorDefault, 0);
        CFStringRef serialRef = (CFStringRef)IORegistryEntryCreateCFProperty(
            service, CFSTR("USB Serial Number"), kCFAllocatorDefault, 0);

        if (vidRef) { CFNumberGetValue(vidRef, kCFNumberIntType,    &vid); CFRelease(vidRef); }
        if (pidRef) { CFNumberGetValue(pidRef, kCFNumberIntType,    &pid); CFRelease(pidRef); }
        if (locRef) { CFNumberGetValue(locRef, kCFNumberSInt32Type, &loc); CFRelease(locRef); }

        const char *name = nameRef
            ? CFStringGetCStringPtr(nameRef, kCFStringEncodingUTF8) : "(unknown)";
        if (!name) name = "(unknown)";

        const char *serial = NULL;
        if (serialRef) {
            serial = CFStringGetCStringPtr(serialRef, kCFStringEncodingUTF8);
            if (serial && strlen(serial) == 0) serial = NULL;
        }

        printf("0x%04x   0x%04x   0x%08x   %-32s %s\n",
               vid, pid, loc, name,
               serial ? serial : "-- NO SERIAL --");

        if (serial) have_serial++;
        total++;

        if (nameRef)   CFRelease(nameRef);
        if (serialRef) CFRelease(serialRef);
        IOObjectRelease(service);
    }

    IOObjectRelease(iterator);

    printf("\n%d device(s) found, %d have a real iSerialNumber, %d do not.\n",
           total, have_serial, total - have_serial);

    } /* autoreleasepool */
    return 0;
}
